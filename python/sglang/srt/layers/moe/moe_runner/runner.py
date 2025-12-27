from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.moe.moe_runner.base import (
    FusedOpPool,
    MoeRunnerConfig,
    PermuteMethodPool,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore
from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsRunnerCore
from sglang.srt.layers.moe.utils import get_moe_a2a_backend

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

logger = logging.getLogger(__name__)


class MoeRunner:

    def __init__(self, runner_backend: MoeRunnerBackend, config: MoeRunnerConfig):
        self.runner_backend = runner_backend
        self.config = config

        self.fused_func = None

        if runner_backend.is_triton():
            self.runner_core = TritonRunnerCore(config)
        elif runner_backend.is_triton_kernels():
            self.runner_core = TritonKernelsRunnerCore(config)
        elif runner_backend.is_deep_gemm():
            self.runner_core = DeepGemmRunnerCore(config)
        elif runner_backend.is_marlin():
            self.runner_core = None  # Marlin only supports fused path
        else:
            raise NotImplementedError(f"Unsupported runner backend: {runner_backend}")

        a2a_backend_name = get_moe_a2a_backend().value
        runner_backend_name = runner_backend.value

        # TODO(cwan): add a server argument to disable fused func
        self.fused_func = FusedOpPool.get_fused_func(
            a2a_backend_name, runner_backend_name
        )

        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        SGLANG_CI_DISABLE_MOE_FUSED_FUNC = os.environ.get(
            "SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "0"
        )
        if SGLANG_CI_DISABLE_MOE_FUSED_FUNC == "1":
            logger.info(
                "SGLANG_CI_DISABLE_MOE_FUSED_FUNC is set to 1, disabling fused func"
            )
            self.fused_func = None

    # Runkai's Remark #46: MoeRunner.run - Execute MoE expert computation.
    # This is called from UnquantizedFusedMoEMethod.forward_cuda (Remark #45).
    # This method orchestrates the MoE computation by either:
    #   1. Using a fused function (optimized single-kernel path) - DEFAULT for Mixtral-8x7B
    #   2. Using separate permute + runner_core + unpermute steps (multi-step path)
    #
    # Call chain for Mixtral-8x7B:
    # 1. UnquantizedFusedMoEMethod.forward_cuda (unquant.py:481): self.runner.run(dispatch_output, quant_info)
    # 2. MoeRunner.run (THIS FUNCTION): routes to self.fused_func
    # 3. fused_experts_none_to_triton (Remark #35 in triton.py:328-357): registered via @register_fused_func
    #
    # Input:
    #   - dispatch_output: StandardDispatchOutput containing:
    #       * hidden_states: [num_tokens, hidden_dim] - Token embeddings to process
    #       * topk_output: StandardTopKOutput with topk_weights [num_tokens, 2] and topk_ids [num_tokens, 2]
    #   - quant_info: TritonMoeQuantInfo containing expert weights (w13_weight, w2_weight)
    # Output:
    #   - CombineInput (StandardCombineInput): hidden_states [num_tokens, hidden_dim] - Expert outputs weighted and combined
    def run(
        self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo
    ) -> CombineInput:

        # Runkai's Remark #47: Check if fused function is available - DEFAULT PATH for Mixtral-8x7B.
        # self.fused_func is set during __init__ (line 49-51) by calling FusedOpPool.get_fused_func.
        # For Mixtral-8x7B with --tp 8:
        #   - a2a_backend_name = "none" (no all-to-all communication for standard TP mode)
        #   - runner_backend_name = "triton" (using Triton MOE backend)
        #   - FusedOpPool.get_fused_func("none", "triton") returns fused_experts_none_to_triton
        #
        # fused_experts_none_to_triton is registered via decorator (triton.py:328):
        #   @register_fused_func("none", "triton")
        #   def fused_experts_none_to_triton(...)
        # This decorator (base.py:212-230) adds the function to FusedOpPool._fused_funcs dict.
        #
        # The fused function is an optimized end-to-end kernel that combines:
        #   - Token permutation based on expert assignment
        #   - Expert computation (fused_moe_triton)
        #   - Result unpermutation and weighted combination
        # into a single efficient operation.
        if self.fused_func is not None:
            # Runkai's Remark #48: Call the fused function directly.
            # For Mixtral-8x7B, this calls fused_experts_none_to_triton (Remark #35).
            #
            # Function signature:
            #   fused_experts_none_to_triton(
            #       dispatch_output: StandardDispatchOutput,
            #       quant_info: TritonMoeQuantInfo,
            #       config: MoeRunnerConfig
            #   ) -> StandardCombineInput
            #
            # Inside fused_experts_none_to_triton (triton.py:328-357):
            # 1. Extract inputs: hidden_states, topk_weights, topk_ids
            # 2. Permute tokens according to expert assignment (group by expert)
            # 3. Call fused_moe_triton to compute expert outputs in batched manner
            # 4. Un-permute results and multiply by topk_weights
            # 5. Return StandardCombineInput with combined hidden_states
            #
            # Example for Mixtral-8x7B with 4 tokens:
            # Input: hidden_states [4, 4096], topk_weights [4, 2], topk_ids [4, 2]
            # → fused_moe_triton computes expert outputs
            # Output: StandardCombineInput(hidden_states=[4, 4096])
            return self.fused_func(dispatch_output, quant_info, self.config)

        dispatch_format = dispatch_output.format.value
        runner_format = self.runner_core.runner_backend.value
        self.pre_permute_func = PermuteMethodPool.get_pre_permute(
            dispatch_format, runner_format
        )

        running_state = {}
        if self.down_gemm_overlap_args is not None:
            running_state["down_gemm_overlap_args"] = self.down_gemm_overlap_args
        if self.meta_overlap_args is not None:
            running_state["meta_overlap_args"] = self.meta_overlap_args

        runner_input = self.pre_permute_func(
            dispatch_output, quant_info, self.config, running_state
        )
        runner_output = self.runner_core.run(runner_input, quant_info, running_state)

        runner_format = self.runner_core.runner_backend.value
        combine_format = dispatch_output.format.value
        self.post_permute_func = PermuteMethodPool.get_post_permute(
            runner_format, combine_format
        )
        combine_input = self.post_permute_func(
            runner_output, quant_info, self.config, running_state
        )

        return combine_input

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        assert self.fused_func is None, "Fused func is not supported for overlap args"
        self.down_gemm_overlap_args = down_gemm_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        assert self.fused_func is None, "Fused func is not supported for overlap args"
        self.down_gemm_overlap_args = None
        self.meta_overlap_args = None
