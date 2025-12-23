from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_local_dp_buffer,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.utils.common import get_bool_env_var, is_hip, is_sm120_supported

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


try:
    if is_sm120_supported():
        from flashinfer import fp4_quantize
    else:
        from sgl_kernel import scaled_fp4_quant as fp4_quantize

    from flashinfer import fp4_quantize as fp4_quantize_flashinfer
except ImportError:
    fp4_quantize = None


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):

    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.num_experts = moe_runner_config.num_experts
        self.num_local_shared_experts = moe_runner_config.num_fused_shared_experts
        self.num_local_routed_experts = (
            moe_runner_config.num_local_experts - self.num_local_shared_experts
        )
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.local_expert_mapping = None

    # Runkai's Remark #27: StandardDispatcher.dispatch() - Route tokens to their selected experts.
    # This is THE dispatch method called from layer.py:947 (Remark #26) for your Mixtral-8x7B --tp 8 setup.
    # Input:
    #   - hidden_states: torch.Tensor [num_tokens, hidden_dim] - Token embeddings (e.g., [4, 4096] for 4 tokens)
    #   - topk_output: TopKOutput - Output from select_experts (Remark #16) containing:
    #       topk_weights [num_tokens, top_k] and topk_ids [num_tokens, top_k]
    # Output: StandardDispatchOutput containing:
    #   - hidden_states: Prepared token embeddings (potentially quantized or all-gathered)
    #   - hidden_states_scale: Scale factors for quantization (None for your FP16/BF16 case)
    #   - topk_output: Modified topk_output with expert ID mapping applied
    #
    # This function performs pre-processing before expert computation:
    #   1. Optional FP4 quantization and all-gather for FlashInfer backends
    #   2. Expert ID mapping for expert parallelism (EP) mode
    #   3. Packaging data for efficient expert processing
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:

        # Runkai's Remark #28: Branch for FlashInfer CUTLASS MOE with FP4 quantization.
        # For Mixtral-8x7B with --tp 8, should_use_flashinfer_cutlass_moe_fp4_allgather() returns False.
        # This branch is only taken when using FlashInfer backend with FP4 quantization.
        # It quantizes hidden_states to FP4 format, performs all-gather across TP ranks,
        # then swizzles the scale factors for optimal memory layout.
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # all-gather fp4 hidden states
            from flashinfer import nvfp4_block_scale_interleave

            global_scale = self.quant_config.get("input_global_scale", None)
            assert global_scale is not None, "input_global_scale is not set"
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            # Quantize before comm, swizzle after.
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                if hidden_states.shape[0] > 0:
                    x, x_sf = fp4_quantize_flashinfer(
                        hidden_states, global_scale, is_sf_swizzled_layout=False
                    )
                else:
                    x_col = hidden_states.shape[1]
                    x = torch.zeros(
                        0, x_col // 2, dtype=torch.uint8, device=hidden_states.device
                    )
                    x_sf = torch.zeros(
                        0, x_col // 16, dtype=torch.uint8, device=hidden_states.device
                    )
            topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
            )
            x_sf = nvfp4_block_scale_interleave(x_sf)

            hidden_states = x
            hidden_states_scale = x_sf
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=topk_output.router_logits,  # never tested
            )
        # Runkai's Remark #29: Standard path for non-quantized inference (your Mixtral-8x7B case).
        # For Mixtral-8x7B with --tp 8, this branch is taken.
        # hidden_states remain unchanged (no quantization).
        # hidden_states_scale is None (no scaling needed for FP16/BF16).
        else:
            hidden_states = hidden_states
            hidden_states_scale = None

        # Runkai's Remark #30: Create expert ID mapping for expert parallelism (EP) mode.
        # For your Mixtral-8x7B with --tp 8 (tensor parallelism only), moe_ep_size=1, so this is SKIPPED.
        #
        # In EP mode (moe_ep_size > 1), experts are distributed across different GPUs:
        # - Each GPU holds a subset of experts (num_local_routed_experts)
        # - local_expert_mapping maps global expert IDs [0, num_experts) to local IDs on this GPU
        # - Experts not on this GPU are mapped to -1
        #
        # Example for 8 experts across 2 GPUs in EP mode:
        #   GPU 0 (moe_ep_rank=0): local_expert_mapping = [0, 1, 2, 3, -1, -1, -1, -1]
        #   GPU 1 (moe_ep_rank=1): local_expert_mapping = [-1, -1, -1, -1, 0, 1, 2, 3]
        # Shared experts (if any) are replicated on all GPUs and mapped at the end.
        if (
            self.moe_ep_size > 1
            and not self.enable_flashinfer_cutlass_moe
            and TopKOutputChecker.format_is_standard(topk_output)
        ):
            # Runkai's Remark #31: Initialize local_expert_mapping on first call.
            # This creates a mapping tensor initialized to -1 (invalid expert).
            # Then fills in the local expert IDs for experts hosted on this GPU.
            if self.local_expert_mapping is None:
                self.local_expert_mapping = torch.full(
                    (self.num_experts,), -1, dtype=torch.int32, device="cuda"
                )
                # Map global expert IDs to local expert IDs for routed experts on this GPU.
                # Range: [moe_ep_rank * num_local_routed_experts : (moe_ep_rank+1) * num_local_routed_experts]
                self.local_expert_mapping[
                    self.moe_ep_rank
                    * self.num_local_routed_experts : (self.moe_ep_rank + 1)
                    * self.num_local_routed_experts
                ] = torch.arange(
                    0, self.num_local_routed_experts, dtype=torch.int32, device="cuda"
                )

                # Runkai's Remark #32: Map shared experts (if any) to local IDs.
                # Shared experts are always placed at the end of the global expert list.
                # They are replicated on all GPUs and mapped to local IDs after routed experts.
                # Example: with 8 routed + 1 shared expert, shared expert global ID 8 → local ID 8 on all GPUs.
                if self.num_local_shared_experts > 0:
                    self.local_expert_mapping[-self.num_local_shared_experts :] = (
                        torch.arange(
                            self.num_local_routed_experts,
                            self.num_local_routed_experts
                            + self.num_local_shared_experts,
                            dtype=torch.int32,
                            device="cpu",
                        )
                    )

        # Runkai's Remark #33: Apply expert ID mapping to convert global IDs to local IDs.
        # For your Mixtral-8x7B with --tp 8, local_expert_mapping is None (no EP), so this is SKIPPED.
        # _use_aiter is False for NVIDIA GPUs (see topk.py:72).
        #
        # In EP mode, this performs the actual mapping:
        #   topk_ids with global expert IDs → local expert IDs using local_expert_mapping
        # Example: topk_ids [[3, 7], [1, 5]] on GPU 1 with mapping [-1,-1,-1,-1,0,1,2,3]
        #   becomes [[0, 3], [-1, 1]] (experts 3,7 → local 0,3; expert 1 not on GPU 1 → -1)
        #
        # For TritonKernelTopKOutput format, this mapping is not yet implemented.
        if self.local_expert_mapping is not None and not _use_aiter:
            if TopKOutputChecker.format_is_standard(topk_output):
                topk_output = topk_output._replace(
                    topk_ids=self.local_expert_mapping[topk_output.topk_ids]
                )
            elif TopKOutputChecker.format_is_triton_kernels(topk_output):
                raise NotImplementedError()

        # Runkai's Remark #34: Return StandardDispatchOutput with prepared data.
        # For Mixtral-8x7B with --tp 8, this returns:
        #   - hidden_states: [num_tokens, hidden_dim] - unchanged token embeddings
        #   - hidden_states_scale: None (no quantization)
        #   - topk_output: StandardTopKOutput with topk_weights, topk_ids (unchanged, no EP mapping)
        #
        # This output is then passed to run_moe_core (layer.py:940) which calls the quantization
        # method's apply() function to compute expert outputs using the prepared hidden states
        # and routing information (topk_weights, topk_ids).
        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
            get_tp_group().reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
        return hidden_states
