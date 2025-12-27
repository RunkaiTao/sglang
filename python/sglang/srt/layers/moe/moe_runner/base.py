from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, TypeGuard

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonRunnerCore,
        TritonRunnerInput,
        TritonRunnerOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        CombineInputFormat,
        DispatchOutput,
        DispatchOutputFormat,
    )


@dataclass
class MoeRunnerConfig:
    # MoE parameters
    num_experts: Optional[int] = None
    num_local_experts: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size_per_partition: Optional[int] = None
    layer_id: Optional[int] = None
    top_k: Optional[int] = None
    num_fused_shared_experts: Optional[int] = None
    params_dtype: Optional[torch.dtype] = None

    # Runner configuration
    activation: str = "silu"
    is_gated: bool = True
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None


@dataclass
class RunnerInput(ABC):
    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerInput]:
        return self.runner_backend == MoeRunnerBackend.TRITON


class RunnerOutput(ABC):
    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerOutput]:
        return self.runner_backend == MoeRunnerBackend.TRITON


@dataclass
class MoeQuantInfo(ABC):
    """Moe quantization data."""

    pass


class MoeRunnerCore(ABC):
    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    @abstractmethod
    def run(
        self, runner_input: RunnerInput, quant_info: MoeQuantInfo, running_state: dict
    ) -> RunnerOutput:
        pass

    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerCore]:
        return self.runner_backend == MoeRunnerBackend.TRITON


# Runkai's Remark #49: FusedOpPool - Registry for optimized fused MoE functions.
# This class maintains a global registry of fused functions that combine multiple MoE operations
# (token permutation, expert computation, result combination) into single optimized kernels.
# The registry maps (a2a_backend, runner_backend) pairs to their corresponding fused functions.
#
# For Mixtral-8x7B with --tp 8:
#   - Key: ("none", "triton") → fused_experts_none_to_triton (Remark #35)
#   - "none": no all-to-all communication (standard tensor parallelism)
#   - "triton": uses Triton MOE backend for expert computation
#
# The registration happens at MODULE IMPORT TIME via decorators (see Remark #50).
class FusedOpPool:
    # Runkai's Remark #50: Class-level dictionary storing all registered fused functions.
    # Structure: _fused_funcs[(a2a_backend_name, runner_backend_name)] = fused_function
    # This is populated when Python imports the moe_runner modules.
    #
    # Registration timeline for Mixtral-8x7B:
    # 1. Python imports sglang.srt.layers.moe.moe_runner.triton module
    # 2. Module contains: @register_fused_func("none", "triton") at line 362
    # 3. Decorator executes immediately, calling FusedOpPool.register_fused_func (THIS METHOD)
    # 4. fused_experts_none_to_triton is added to _fused_funcs[("none", "triton")]
    # 5. Later, MoeRunner.__init__ retrieves it via get_fused_func (Remark #52)
    #
    # Example registered functions:
    #   - ("none", "triton") → fused_experts_none_to_triton (Mixtral, standard MoE)
    #   - ("none", "marlin") → fused_experts_none_to_marlin (quantized models)
    _fused_funcs: dict[str, Callable] = {}

    # Runkai's Remark #51: Register a fused function for given backend combination.
    # This method is called by the @register_fused_func decorator (defined at line 212).
    # It's NOT called directly in code - it's invoked automatically when Python imports modules.
    #
    # Call chain for Mixtral-8x7B registration:
    # 1. Python import: from sglang.srt.layers.moe.moe_runner.triton import ...
    # 2. Decorator @register_fused_func("none", "triton") executes (triton.py:362)
    # 3. Decorator function (line 212-230) calls THIS METHOD: cls.register_fused_func(...)
    # 4. This method stores fused_experts_none_to_triton in _fused_funcs dictionary
    #
    # Parameters:
    #   - a2a_backend_name: All-to-all communication backend ("none" for Mixtral --tp 8)
    #   - runner_backend_name: MoE computation backend ("triton" for Mixtral)
    #   - fused_func: The actual function to register (e.g., fused_experts_none_to_triton)
    @classmethod
    def register_fused_func(
        cls, a2a_backend_name: str, runner_backend_name: str, fused_func: Callable
    ):
        key = (a2a_backend_name, runner_backend_name)
        if key in cls._fused_funcs:
            raise ValueError(
                f"Fused function for {a2a_backend_name} to {runner_backend_name} is already registered."
            )
        assert MoeA2ABackend(
            a2a_backend_name
        ), f"Invalid dispatch name: {a2a_backend_name}"
        assert MoeRunnerBackend(
            runner_backend_name
        ), f"Invalid runner name: {runner_backend_name}"
        # Store the fused function in the class-level registry
        # For Mixtral: _fused_funcs[("none", "triton")] = fused_experts_none_to_triton
        cls._fused_funcs[key] = fused_func

    # Runkai's Remark #52: Retrieve a registered fused function by backend names.
    # This is called during MoeRunner initialization (runner.py:49-51).
    #
    # Call chain for Mixtral-8x7B:
    # 1. UnquantizedFusedMoEMethod.__init__ creates MoeRunner (unquant.py initialization)
    # 2. MoeRunner.__init__ (runner.py:28): calls self.fused_func = FusedOpPool.get_fused_func(...)
    # 3. THIS METHOD: looks up _fused_funcs[("none", "triton")]
    # 4. Returns: fused_experts_none_to_triton (which was registered at module import time)
    # 5. MoeRunner stores it as self.fused_func for later use in run() (Remark #47)
    #
    # Parameters:
    #   - dispatch_name: A2A backend name ("none" for standard TP)
    #   - runner_name: Runner backend name ("triton" for Mixtral)
    # Returns:
    #   - The registered fused function, or None if not found
    #   - For Mixtral: returns fused_experts_none_to_triton
    @classmethod
    def get_fused_func(cls, dispatch_name: str, runner_name: str) -> Optional[Callable]:
        key = (dispatch_name, runner_name)
        fused_func = cls._fused_funcs.get(key)
        return fused_func


class PermuteMethodPool:
    _pre_permute_methods: dict[
        Tuple[DispatchOutputFormat, MoeRunnerBackend], Callable
    ] = {}
    _post_permute_methods: dict[
        Tuple[MoeRunnerBackend, CombineInputFormat], Callable
    ] = {}

    @classmethod
    def register_pre_permute(
        cls,
        dispatch_output_name: str,
        runner_backend_name: str,
        permute_func: Callable,
    ):
        """
        Register a customized pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.

        :param dispatch_output_name: The DispatchOutputFormat name.
        :param runner_backend_name: The MoeRunnerBackend name.
        :param permute_func: The permute function to register.
        """
        # TODO: check if registration is valid
        key = (dispatch_output_name, runner_backend_name)
        if key in cls._pre_permute_methods:
            raise ValueError(
                f"Pre-permute method for {dispatch_output_name} to {runner_backend_name} is already registered."
            )
        cls._pre_permute_methods[key] = permute_func

    @classmethod
    def register_post_permute(
        cls,
        runner_backend_name: str,
        combine_input_name: str,
        permute_func: Callable,
    ):
        """
        Register a customized post-permute function for the given MoeRunnerBackend and CombineInputFormat.

        :param runner_backend_name: The MoeRunnerBackend name.
        :param combine_input_name: The CombineInputFormat name.
        :param permute_func: The permute function to register.
        """
        # TODO: check if registration is valid
        key = (runner_backend_name, combine_input_name)
        if key in cls._post_permute_methods:
            raise ValueError(
                f"Post-permute method for {runner_backend_name} to {combine_input_name} is already registered."
            )
        cls._post_permute_methods[key] = permute_func

    @classmethod
    def get_pre_permute(
        cls,
        dispatch_output_format: DispatchOutputFormat,
        runner_input_format: MoeRunnerBackend,
    ) -> Callable:
        """
        Retrieve the pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.

        :param dispatch_output_format: The DispatchOutputFormat type.
        :param runner_input_format: The MoeRunnerBackend type.
        :return: The registered permute function or None if not found.
        """
        key = (dispatch_output_format, runner_input_format)
        pre_permute_func = cls._pre_permute_methods.get(key)
        assert (
            pre_permute_func is not None
        ), f"Pre-permute function for {dispatch_output_format} to {runner_input_format} is not registered"
        return pre_permute_func

    @classmethod
    def get_post_permute(
        cls,
        runner_output_format: MoeRunnerBackend,
        combine_input_format: CombineInputFormat,
    ) -> Callable:
        """
        Retrieve the post-permute function for the given MoeRunnerBackend and CombineInputFormat.

        :param runner_output_format: The MoeRunnerBackend type.
        :param combine_input_format: The CombineInputFormat type.
        :return: The registered permute function or None if not found.
        """
        key = (runner_output_format, combine_input_format)
        post_permute_func = cls._post_permute_methods.get(key)
        assert (
            post_permute_func is not None
        ), f"Post-permute function for {runner_output_format} to {combine_input_format} is not registered"
        return post_permute_func


# Runkai's Remark #53: register_fused_func - Decorator factory for registering fused MoE functions.
# This is a PARAMETRIZED DECORATOR that takes backend names and returns a decorator function.
# It's used to automatically register fused functions at module import time.
#
# Python decorator pattern explanation:
# 1. @register_fused_func("none", "triton") is called FIRST with parameters
# 2. This function RETURNS the inner decorator function
# 3. The returned decorator is applied to fused_experts_none_to_triton
# 4. The decorator calls FusedOpPool.register_fused_func to store the function
# 5. The decorator returns the original function unchanged
#
# Example usage in triton.py:362:
#   @register_fused_func("none", "triton")
#   def fused_experts_none_to_triton(...):
#       ...
#
# Execution flow for Mixtral-8x7B:
# 1. Python imports sglang.srt.layers.moe.moe_runner.triton
# 2. Encounters decorator: @register_fused_func("none", "triton")
# 3. Calls THIS FUNCTION with a2a_backend_name="none", runner_backend_name="triton"
# 4. THIS FUNCTION returns the inner decorator function
# 5. Inner decorator receives fused_experts_none_to_triton as argument
# 6. Inner decorator registers it in FusedOpPool (Remark #51)
# 7. Function is now available via FusedOpPool.get_fused_func("none", "triton") (Remark #52)
#
# Parameters:
#   - a2a_backend_name: All-to-all communication backend ("none" for standard TP, "a2a" for EP)
#   - runner_backend_name: MoE computation backend ("triton", "marlin", etc.)
# Returns:
#   - decorator: Inner function that performs the actual registration
def register_fused_func(
    a2a_backend_name: str,
    runner_backend_name: str,
) -> Callable:
    """
    Decorator to register a fused function for the given DispatchOutputFormat and MoeRunnerBackend.

    :param a2a_backend_name: The A2A backend name.
    :param runner_backend_name: The MoeRunnerBackend name.
    :return: The decorator function.
    """

    # Runkai's Remark #54: Inner decorator function - Performs actual registration.
    # This is the function returned by register_fused_func that wraps the target function.
    # It's called with the decorated function (e.g., fused_experts_none_to_triton) as argument.
    #
    # HOW DECORATORS WORK IN PYTHON:
    # When Python sees:
    #   @register_fused_func("none", "triton")
    #   def fused_experts_none_to_triton(...):
    #       ...
    #
    # It executes (at module import time):
    #   temp_decorator = register_fused_func("none", "triton")  # Returns THIS FUNCTION
    #   fused_experts_none_to_triton = temp_decorator(fused_experts_none_to_triton)
    #
    # Execution sequence:
    # 1. THIS FUNCTION receives fused_func (the decorated function) as parameter
    # 2. Calls FusedOpPool.register_fused_func (Remark #51) to store in registry
    # 3. Returns the original fused_func unchanged (so it can still be called normally)
    #
    # For Mixtral-8x7B:
    # - fused_func = fused_experts_none_to_triton
    # - a2a_backend_name = "none" (from outer scope)
    # - runner_backend_name = "triton" (from outer scope)
    # - Result: FusedOpPool._fused_funcs[("none", "triton")] = fused_experts_none_to_triton
    #
    # Parameter:
    #   - fused_func: The function being decorated (e.g., fused_experts_none_to_triton)
    # Returns:
    #   - fused_func: The same function unchanged (registration is a side effect)
    def decorator(fused_func: Callable):
        # Runkai's Remark #55: Register the fused function in FusedOpPool.
        # This calls the class method documented in Remark #51.
        # The registration happens as a SIDE EFFECT during module import.
        #
        # For Mixtral-8x7B execution:
        # - a2a_backend_name = "none" (captured from outer scope)
        # - runner_backend_name = "triton" (captured from outer scope)
        # - fused_func = fused_experts_none_to_triton
        # - This adds entry: _fused_funcs[("none", "triton")] = fused_experts_none_to_triton
        #
        # TIMING: This executes at MODULE IMPORT TIME, not when functions are called.
        # By the time MoeRunner.__init__ runs, this registration has already happened.
        FusedOpPool.register_fused_func(
            a2a_backend_name, runner_backend_name, fused_func
        )
        # Runkai's Remark #56: Return the original function unchanged.
        # This allows the function to be used normally (e.g., called directly for testing).
        # The decorator is "transparent" - it doesn't modify the function's behavior.
        #
        # After this returns, fused_experts_none_to_triton is:
        # 1. Stored in FusedOpPool._fused_funcs[("none", "triton")] (for registry lookup)
        # 2. Still available as triton.fused_experts_none_to_triton (for direct calls)
        return fused_func

    return decorator


def register_pre_permute(
    dispatch_output_name: str,
    runner_backend_name: str,
) -> Callable:
    """
    Decorator to register a pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.

    :param dispatch_output_name: The DispatchOutputFormat name.
    :param runner_backend_name: The MoeRunnerBackend name.
    :return: The decorator function.
    """

    def decorator(
        permute_func: Callable[
            [DispatchOutput, MoeQuantInfo, MoeRunnerConfig, dict], RunnerInput
        ],
    ) -> Callable:
        PermuteMethodPool.register_pre_permute(
            dispatch_output_name, runner_backend_name, permute_func
        )
        return permute_func

    return decorator


def register_post_permute(
    runner_backend_name: str,
    combine_input_name: str,
) -> Callable:
    """
    Decorator to register a post-permute function for the given MoeRunnerBackend and CombineInputFormat.

    :param runner_backend_name: The MoeRunnerBackend name.
    :param combine_input_name: The CombineInputFormat name.
    :return: The decorator function.
    """

    def decorator(
        permute_func: Callable[
            [RunnerOutput, MoeQuantInfo, MoeRunnerConfig, dict], CombineInput
        ],
    ) -> Callable:
        PermuteMethodPool.register_post_permute(
            runner_backend_name, combine_input_name, permute_func
        )
        return permute_func

    return decorator
