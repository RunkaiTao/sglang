# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

import torch

try:
    from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx, routing
except ImportError:
    pass

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.layers.quantization import QuantizationConfig


logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    from sgl_kernel import moe_fused_gate

    try:
        from sgl_kernel import kimi_k2_moe_fused_gate
    except ImportError as e:
        pass

if _is_cuda or _is_hip:
    from sgl_kernel import topk_softmax

    try:
        from sgl_kernel import topk_sigmoid
    except ImportError:
        pass
if _use_aiter:
    try:
        from aiter import biased_grouped_topk as aiter_biased_grouped_topk
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")

# -------------------------------- TopKConfig ---------------------------------------


@dataclass
class TopKConfig:
    top_k: int
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    fused_shared_experts_scaling_factor: Optional[float] = None
    output_format: Optional[TopKOutputFormat] = None
    scoring_func: str = "softmax"


# -------------------------------- TopKOutput ---------------------------------------


class TopKOutputChecker:

    @staticmethod
    def format_is_standard(topk_output: TopKOutput) -> TypeGuard[StandardTopKOutput]:
        return isinstance(topk_output, StandardTopKOutput)

    @staticmethod
    def format_is_triton_kernels(
        topk_output: TopKOutput,
    ) -> TypeGuard[TritonKernelTopKOutput]:
        return isinstance(topk_output, TritonKernelTopKOutput)

    @staticmethod
    def format_is_bypassed(topk_output: TopKOutput) -> TypeGuard[BypassedTopKOutput]:
        return isinstance(topk_output, BypassedTopKOutput)


class TopKOutputFormat(IntEnum):
    STANDARD = auto()
    TRITON_KERNEL = auto()
    BYPASSED = auto()


@runtime_checkable
class TopKOutput(Protocol):
    """Protocol for top-k outputs in different formats."""

    @property
    def format(self) -> TopKOutputFormat:
        """The format of the output."""
        ...


class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


class TritonKernelTopKOutput(NamedTuple):
    """Triton kernel top-k output format."""

    routing_data: RoutingData
    gather_indx: GatherIndx
    scatter_indx: ScatterIndx

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.TRITON_KERNEL


class BypassedTopKOutput(NamedTuple):
    """Bypassed top-k output format."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: Optional[torch.Tensor] = None
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED


# -------------------------------- TopK ---------------------------------------


class TopK(CustomOp):
    """
    Parameters:
    --top_k: The all number of top experts selected per token, including the fused shared expert(s).
    --num_fused_shared_experts: num of shared experts, can be activate both in TP or EP mode.
    --routed_scaling_factor: the scaling factor for routed experts in topk_weights.
    --fused_shared_experts_scaling_factor: scaling factor for fused shared experts on AMD-platform.
    """

    def __init__(
        self,
        top_k: int,
        *,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        quant_config: Optional[QuantizationConfig] = None,
        routed_scaling_factor: Optional[float] = None,
        apply_routed_scaling_factor_on_output: Optional[bool] = False,
        output_format: Optional[TopKOutputFormat] = None,
        fused_shared_experts_scaling_factor: Optional[float] = None,
    ):
        # NOTE: scoring_func is not used for now, but we keep it for future use
        # see https://github.com/sgl-project/sglang/pull/4505 for more details
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None

        self.topk_config = TopKConfig(
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            output_format=output_format,
            scoring_func=scoring_func,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        self.topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    # Runkai's Remark #1: This is the forward_cuda method of the TopK class.
    # It handles the top-k expert selection computation specifically for CUDA devices (NVIDIA GPUs).
    # Input:
    #   - hidden_states: torch.Tensor - The input token embeddings/features (shape: [num_tokens, hidden_dim])
    #   - router_logits: torch.Tensor - Raw scores from the router network indicating affinity to each expert (shape: [num_tokens, num_experts])
    #   - num_token_non_padded: Optional[torch.Tensor] - Number of non-padded tokens (used for batched inference)
    #   - expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] - Information for expert placement across devices in distributed setting
    # Output: TopKOutput - One of three formats: StandardTopKOutput, TritonKernelTopKOutput, or BypassedTopKOutput
    #
    # This function determines which output format to use based on the MOE backend configuration,
    # then dispatches to the appropriate computation path.
    # Example: For Mixtral-8x7B with hidden_states [4, 4096] and router_logits [4, 8], top_k=2
    #          returns StandardTopKOutput with topk_weights [4, 2] and topk_ids [4, 2]
    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        # Runkai's Remark #2: Determine the output format for the top-k selection.
        # This checks if a specific output format is configured in topk_config.
        # If not specified, it automatically selects based on the MOE runner backend being used.
        # The three possible formats are:
        #   - TRITON_KERNEL: Uses Triton-based kernels for efficient routing
        #   - BYPASSED: Defers the top-k computation to later in the pipeline (used by FlashInfer backends)
        #   - STANDARD: Traditional format with topk_weights and topk_ids tensors
        if self.topk_config.output_format is not None:
            output_format = self.topk_config.output_format
        elif get_moe_runner_backend().is_triton_kernels():
            output_format = TopKOutputFormat.TRITON_KERNEL
        elif (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_mxfp4()
        ):
            output_format = TopKOutputFormat.BYPASSED
        else:
            output_format = TopKOutputFormat.STANDARD

        # Runkai's Remark #3: Handle TRITON_KERNEL output format.
        # This path uses the Triton kernel library's 'routing' function.
        # The sm_first parameter controls whether softmax is applied before or after selection:
        #   - sm_first=False (when renormalize=True): Apply softmax, then select top-k, then renormalize
        #   - sm_first=True (when renormalize=False): Select top-k first, then apply softmax
        # Returns: TritonKernelTopKOutput containing routing_data, gather_idx, and scatter_idx
        # These indices are used by Triton kernels for efficient expert computation and result gathering.
        if output_format == TopKOutputFormat.TRITON_KERNEL:
            # renormalize=True is equivalent to sm_first=False
            routing_data, gather_idx, scatter_idx = routing(
                router_logits,
                self.topk_config.top_k,
                sm_first=not self.topk_config.renormalize,
            )
            return TritonKernelTopKOutput(routing_data, gather_idx, scatter_idx)
        # Runkai's Remark #4: Handle BYPASSED output format.
        # In this mode, the top-k selection is deferred - the function simply packages
        # the inputs (hidden_states, router_logits, config) into a BypassedTopKOutput object.
        # The actual top-k computation will be performed later in the MOE computation pipeline.
        # This is used by FlashInfer backends (flashinfer_trtllm and flashinfer_mxfp4) which
        # integrate the routing and expert computation into a single fused kernel.
        elif output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        # Runkai's Remark #5: Handle STANDARD output format.
        # This is the default path that performs actual top-k expert selection.
        # It sets torch_native=False to use optimized CUDA kernels instead of PyTorch native ops.
        # The computation is wrapped in use_symmetric_memory context manager for efficient
        # memory allocation in tensor-parallel distributed settings.
        # Calls select_experts() which will dispatch to the appropriate kernel (fused_topk,
        # grouped_topk, or biased_grouped_topk) based on the model architecture.
        # Returns: StandardTopKOutput with topk_weights (shape: [num_tokens, top_k]),
        # topk_ids (shape: [num_tokens, top_k]), and router_logits.
        else:
            self.topk_config.torch_native = False
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                topk_output = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_config=self.topk_config,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                )
            return topk_output

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def forward_npu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:

        from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu

        return fused_topk_npu(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def empty_topk_output(self, device: torch.device) -> TopKOutput:
        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
            topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        # FIXME: router_logits should be of size (0, num_experts)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)


# ------------------------------- TopK implementation -------------------------------------


def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    def scoring_func_impl(gating_output: torch.Tensor) -> torch.Tensor:
        if scoring_func == "softmax":
            return gating_output.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            return gating_output.sigmoid()
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = scoring_func_impl(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = scoring_func_impl(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def apply_topk_weights_cpu(need_apply, topk_weights, inputs):
    if not need_apply:
        return inputs, topk_weights

    # TODO: fuse below processing in fused_experts_cpu kernel
    inputs = inputs * topk_weights.to(inputs.dtype)
    topk_weights = torch.ones_like(
        topk_weights, dtype=torch.float32
    )  # clear topk_weights as already applied

    return inputs, topk_weights


# Runkai's Remark #17: Optimized fused top-k kernel for standard MOE models (including Mixtral-8x7B).
# This is THE KEY FUNCTION called by select_experts (Remark #12) for your Mixtral-8x7B setup.
# Input:
#   - hidden_states: torch.Tensor [num_tokens, hidden_dim] - Token embeddings (not directly used, only for shape validation)
#   - gating_output: torch.Tensor [num_tokens, num_experts] - Router logits from the gate network
#   - topk: int - Number of experts to select per token (2 for Mixtral-8x7B)
#   - renormalize: bool - Whether to normalize weights after selection (True for Mixtral)
#   - correction_bias: Optional[torch.Tensor] - Load balancing bias (None for Mixtral, used by some models)
#   - num_token_non_padded: Optional[torch.Tensor] - Number of real tokens excluding padding
#   - expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] - For expert parallelism (None for --tp 8)
#   - scoring_func: str - "softmax" (default for Mixtral) or "sigmoid" (for some other models)
# Output: (topk_weights, topk_ids) - Selected expert weights and IDs
# Example for Mixtral with 4 tokens and 8 experts:
#   Input: gating_output [4, 8] with raw logits
#   Output: topk_weights [4, 2], topk_ids [4, 2] with top-2 experts per token
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    scoring_func: str = "softmax",
):
    # Runkai's Remark #18: Validate that number of tokens matches between hidden_states and gating_output.
    # This ensures the router logits correspond to the correct tokens.
    # For Mixtral: if hidden_states is [4, 4096], gating_output must be [4, 8].
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    # Runkai's Remark #19: Extract number of tokens (M) from hidden_states shape.
    # M represents the batch size in terms of tokens (not sequences).
    # For Mixtral: M could be 4, 16, 128, etc. depending on batch size and sequence length.
    M, _ = hidden_states.shape

    # Runkai's Remark #20: Allocate output tensors for topk_weights and topk_ids.
    # These are allocated as empty tensors on the same device as hidden_states (GPU for --tp 8).
    # topk_weights: [M, topk] stores the routing weights (probabilities) for selected experts
    # topk_ids: [M, topk] stores the expert indices (0-7 for Mixtral-8x7B)
    # Using float32 for weights and int32 for IDs to match kernel expectations.
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    # Runkai's Remark #21: Dispatch to appropriate scoring function kernel.
    # For Mixtral-8x7B, scoring_func="softmax" is used (default), so topk_softmax is called.
    # topk_softmax is an optimized CUDA kernel from sgl_kernel (imported at line 83).
    # The kernel performs in-place operations:
    #   1. Apply softmax to gating_output: scores = exp(logits) / sum(exp(logits))
    #   2. Select top-k experts with highest scores
    #   3. If renormalize=True: normalize selected weights to sum to 1
    # The results are written directly into topk_weights and topk_ids tensors.
    # Alternative: topk_sigmoid uses sigmoid activation (for models like certain DeepSeek variants).
    if scoring_func == "softmax":
        topk_softmax(
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
        )
    elif scoring_func == "sigmoid":
        topk_sigmoid(
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
            correction_bias,
        )
    else:
        raise ValueError(f"Invalid scoring function: {scoring_func}")

    # Runkai's Remark #22: Convert logical expert IDs to physical expert IDs for expert parallelism.
    # topk_ids_logical_to_physical (from srt/eplb/expert_location_dispatch.py:76) maps expert IDs
    # from logical (model-defined) to physical (device-specific) locations in EP mode.
    # For your Mixtral-8x7B with --tp 8 (tensor parallelism, NOT expert parallelism),
    # expert_location_dispatch_info is None, so topk_ids are returned unchanged.
    # In EP mode, this would map e.g., expert 3 → GPU 1, expert 5 → GPU 2, etc.
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)

    # Runkai's Remark #23: Mask out expert IDs for padded tokens by setting them to -1.
    # _mask_topk_ids_padded_region (defined at line 696) sets topk_ids to -1 for padded positions.
    # In batched inference, sequences may be padded to the same length. This marks padding tokens
    # so they don't route to real experts (saving computation).
    # For single sequence inference or when num_token_non_padded is None, this does nothing.
    # Example: if num_token_non_padded=3 and M=4, then topk_ids[3, :] = -1
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)

    # Runkai's Remark #24: Return the selected expert weights and IDs.
    # topk_weights: [M, topk] - Normalized routing weights for selected experts (sum to 1 per token if renormalize=True)
    # topk_ids: [M, topk] - Expert indices for selected experts
    # For Mixtral-8x7B with M=4 tokens, topk=2:
    #   topk_weights might be [[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]]
    #   topk_ids might be [[3, 7], [1, 5], [2, 6], [0, 4]]
    # These are then used by the MoE layer to compute weighted expert outputs.
    return topk_weights, topk_ids


# This is used by the Deepseek V2/V3/R1 series models
@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert not apply_routed_scaling_factor_on_output
    assert expert_location_dispatch_info is None
    return torch.ops.sgl_kernel.grouped_topk_cpu(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def kimi_k2_biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """
    Optimized version for num_expert_group=1 case (e.g., Kimi K2 with 384 experts).
    Simplifies the grouped topk logic by removing unnecessary group masking operations.
    Note: This function assumes num_fused_shared_experts=0.
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]

    # When num_expert_group=1, no need for group masking
    # Directly compute scores with correction bias
    tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    # Directly select topk experts (no need to sort since num_fused_shared_experts=0)
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(
    topk_ids, expert_location_dispatch_info, num_token_non_padded
):
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids


def biased_grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    # TODO: moe_fused_gate kernel is not supported for num_fused_shared_experts > 0 now.
    if (
        _is_cuda
        and gating_output.shape[1] // num_expert_group
        <= 32  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this assertion.
        and is_power_of_two(correction_bias.shape[0])
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            apply_routed_scaling_factor_on_output,
        )
        # TODO merge into kernel
        if (expert_location_dispatch_info is not None) or (
            num_token_non_padded is not None
        ):
            topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )
        return topk_weights, topk_ids
    elif _use_aiter:
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        token = gating_output.shape[0]
        device = gating_output.device
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch: hidden_states.shape[0] = {hidden_states.shape[0]}, gating_output.shape[0] = {gating_output.shape[0]}"
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        aiter_biased_grouped_topk(
            gating_output,
            correction_bias.to(dtype=gating_output.dtype),
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        )
        return topk_weights, topk_ids
    else:
        # Use optimized path for Kimi K2 (384 experts with num_expert_group=1)
        num_experts = gating_output.shape[1]
        if _is_cuda and num_experts == 384 and num_expert_group == 1:
            return kimi_k2_moe_fused_gate(
                gating_output.to(dtype=torch.float32),
                correction_bias,
                topk=topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            return biased_grouped_topk_impl(
                hidden_states,
                gating_output,
                correction_bias,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )


def biased_grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    compiled: bool = True,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert expert_location_dispatch_info is None
    assert not apply_routed_scaling_factor_on_output, "Not implemented"
    return torch.ops.sgl_kernel.biased_grouped_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    )


if _is_cpu and _is_cpu_amx_available:
    biased_grouped_topk = biased_grouped_topk_cpu
    grouped_topk = grouped_topk_cpu
    fused_topk_native = fused_topk_cpu
    fused_topk = fused_topk_cpu
else:
    biased_grouped_topk = biased_grouped_topk_gpu
    grouped_topk = grouped_topk_gpu
    fused_topk_native = fused_topk_torch_native


# Runkai's Remark #6: Main expert selection function that dispatches to different routing algorithms.
# This is the core function called by forward_cuda (Remark #5) to perform actual top-k expert selection.
# Input:
#   - hidden_states: torch.Tensor [num_tokens, hidden_dim] - Token embeddings from previous layer
#   - router_logits: torch.Tensor [num_tokens, num_experts] - Raw routing scores for each token-expert pair
#   - topk_config: TopKConfig - Configuration object containing all routing parameters
#   - num_token_non_padded: Optional[torch.Tensor] - Scalar indicating number of real (non-padding) tokens
#   - expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] - For distributed expert placement
# Output: StandardTopKOutput with topk_weights, topk_ids, and router_logits
# For Mixtral-8x7B: receives router_logits [num_tokens, 8], returns topk_weights and topk_ids [num_tokens, 2]
def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:

    # Runkai's Remark #7: Extract configuration parameters from topk_config.
    # These values come from the TopKConfig object created during model initialization (see line 227-241).
    # Key parameters:
    #   - top_k: total experts per token (e.g., 2 for Mixtral)
    #   - use_grouped_topk: True for DeepSeek models, False for Mixtral
    #   - renormalize: whether to normalize weights after selection (usually True)
    #   - num_fused_shared_experts: shared experts that process all tokens (0 for Mixtral)
    #   - correction_bias: bias tensor for load balancing (used by DeepSeek V3)
    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )
    fused_shared_experts_scaling_factor = (
        topk_config.fused_shared_experts_scaling_factor
    )
    scoring_func = topk_config.scoring_func

    # Runkai's Remark #8: Transform inputs for expert parallelism (EP) mode.
    # In EP mode, experts are distributed across different devices/nodes.
    # This function (from srt/eplb/expert_location_dispatch.py:64) transforms the router_logits
    # to map logical expert IDs to physical expert locations, and adjusts correction_bias if needed.
    # For standard TP mode (like your Mixtral-8x7B --tp 8), expert_location_dispatch_info is None,
    # so router_logits and correction_bias are returned unchanged.
    router_logits, correction_bias = (
        expert_location_dispatch.transform_select_experts_inputs(
            router_logits=router_logits,
            correction_bias=correction_bias,
            info=expert_location_dispatch_info,
        )
    )

    # Runkai's Remark #9: Calculate number of routed experts (excluding shared experts).
    # Some models (like DeepSeek) have shared experts that always process tokens, plus routed experts.
    # For Mixtral-8x7B: num_fused_shared_experts=0, so num_routed_topk = 2 - 0 = 2
    # For DeepSeek V3 with shared experts: if top_k=8 and num_fused_shared_experts=1, then num_routed_topk=7
    num_routed_topk = top_k - num_fused_shared_experts

    # Runkai's Remark #10: Branch 1 - Grouped top-k routing for DeepSeek V2/V3/R1 models.
    # use_grouped_topk is False for Mixtral-8x7B, so this branch is skipped.
    # For DeepSeek models, experts are organized into groups. First select top groups, then top experts within those groups.
    # Example: 256 experts in 8 groups → select top 2 groups → select top 6 experts from those 2 groups.
    # If correction_bias is None: uses grouped_topk (line 479, softmax-based)
    # If correction_bias exists: uses biased_grouped_topk (line 617, sigmoid-based with load balancing bias)
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
    # Runkai's Remark #11: Branch 2 - PyTorch native implementation (CPU or debugging).
    # This branch is taken when torch_native=True (set by forward_native method at line 251).
    # Uses pure PyTorch operations (softmax, torch.topk) instead of optimized kernels.
    # For Mixtral-8x7B with --tp 8, this branch is NOT taken (torch_native=False in forward_cuda).
    # Used mainly for CPU inference or debugging/verification of custom kernels.
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )
    # Runkai's Remark #12: Branch 3 - Optimized fused_topk kernel (default for Mixtral-8x7B).
    # THIS IS THE PATH TAKEN FOR MIXTRAL-8x7B with your command line.
    # Uses optimized CUDA kernel topk_softmax from sgl_kernel (line 455) for GPU acceleration.
    # The kernel performs: softmax(router_logits) → select top-k → renormalize weights.
    # For Mixtral: receives router_logits [num_tokens, 8] → returns topk_weights, topk_ids [num_tokens, 2]
    # _use_aiter is False (only True for AMD HIP with SGLANG_USE_AITER env var, see line 72)
    # Also used by Qwen3MOE and other standard MOE models.
    elif custom_routing_function is None:
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        # Qwen3MOE uses fused_topk
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            scoring_func=scoring_func,
        )
    # Runkai's Remark #13: Branch 4 - Custom routing function for specialized models.
    # Some models may provide a custom_routing_function in topk_config for non-standard routing logic.
    # For Mixtral-8x7B, custom_routing_function is None, so this branch is not taken.
    # This allows flexibility for research models with novel routing mechanisms.
    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
        )

    # Runkai's Remark #14: Append shared experts when using AMD AITER backend.
    # This code path is ONLY for AMD GPUs with SGLANG_USE_AITER=True (_use_aiter from line 72).
    # For NVIDIA GPUs running Mixtral-8x7B, _use_aiter=False, so this is skipped.
    # Shared experts are experts that process ALL tokens (in addition to routed experts).
    # This appends shared expert IDs to topk_ids and applies scaling to topk_weights.
    # Example: if topk_ids=[tokens x 2] for routed experts and num_fused_shared_experts=1,
    # result would be topk_ids=[tokens x 3] with last column being shared expert ID.
    if num_fused_shared_experts > 0 and _use_aiter:
        M, N = router_logits.shape
        scale_factor = (
            1.0
            if fused_shared_experts_scaling_factor is None
            else fused_shared_experts_scaling_factor
        )

        # Lazy import to avoid circular-import issues
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
            fused_append_shared_experts,
        )

        topk_ids, topk_weights = fused_append_shared_experts(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            scale_factor,
            N,  # base id for shared experts
        )

    # Runkai's Remark #15: Record expert selection statistics for load balancing monitoring.
    # get_global_expert_distribution_recorder() (from srt/eplb/expert_distribution.py) tracks
    # which experts are being selected to detect load imbalance (some experts overused, others underused).
    # This data can be used for adaptive load balancing or profiling.
    # For Mixtral-8x7B, this records the distribution of topk_ids across the 8 experts.
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    # Runkai's Remark #16: Return final output with selected experts and weights.
    # StandardTopKOutput is a NamedTuple (defined at line 152) containing:
    #   - topk_weights: [num_tokens, top_k] - Normalized routing weights for selected experts
    #   - topk_ids: [num_tokens, top_k] - Expert indices (0-7 for Mixtral-8x7B)
    #   - router_logits: [num_tokens, num_experts] - Original logits (kept for potential loss computation)
    # Example for Mixtral with 4 tokens:
    #   topk_weights: [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]]
    #   topk_ids: [[2, 5], [0, 7], [3, 1], [4, 6]]
    # These will be used by the MoE layer to route tokens to experts and combine expert outputs.
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


# Register fake implementations for torch.compile support
if _is_cuda:

    @torch.library.register_fake("sgl_kernel::moe_fused_gate")
    def _moe_fused_gate(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=0,
        apply_routed_scaling_factor_on_output=False,
    ):
        num_rows = input_tensor.shape[0]
        topk_weights = torch.empty(
            (num_rows, topk), dtype=torch.float32, device=input_tensor.device
        )
        topk_ids = torch.empty(
            (num_rows, topk), dtype=torch.int32, device=input_tensor.device
        )
        return topk_weights, topk_ids

    @register_fake_if_exists("sgl_kernel::kimi_k2_moe_fused_gate")
    def _kimi_k2_moe_fused_gate(
        input_tensor,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    ):
        num_rows = input_tensor.shape[0]
        topk_weights = input_tensor.new_empty(
            num_rows,
            topk,
            dtype=torch.float32,
        )
        topk_ids = input_tensor.new_empty(
            num_rows,
            topk,
            dtype=torch.int32,
        )
        return topk_weights, topk_ids
