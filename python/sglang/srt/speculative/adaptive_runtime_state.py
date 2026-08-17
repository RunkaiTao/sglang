from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    topk: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController / StaticSpecRouter."""

    speculative_num_steps: int
    speculative_num_draft_tokens: int
    topk: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        topk: int | None = None,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            cfg_path=config_path,
        )
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._activate(new_step)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)


class StaticSpecRouter:
    """Switches the full ``(num_steps, eagle_topk, num_draft_tokens)`` config by batch
    size from a precomputed static table — no acceptance-EMA search.

    Drop-in for AdaptiveController in the worker's ``adaptive_controller`` slot: it exposes
    ``register`` / ``init_states`` / ``activate_step_by_batch`` / ``on_verify_complete`` with
    the same signatures the worker calls. Unlike the EMA controller (which adapts only
    num_steps for the topk=1 chain), the router pre-builds one runtime state per distinct
    table config and adapts all three params. Because each batch size maps to exactly one
    config, CUDA graphs are captured once per batch size → ~single-config graph memory.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str):
        from sglang.srt.speculative.adaptive_spec_params import StaticSpecTable

        self.worker = worker
        self.table = StaticSpecTable(config_path)
        self._states: dict[tuple[int, int, int], SpecRuntimeState] = {}

    @staticmethod
    def _key(state: SpecRuntimeState) -> tuple[int, int, int]:
        return (
            state.speculative_num_steps,
            state.topk,
            state.speculative_num_draft_tokens,
        )

    def _worker_config(self) -> tuple[int, int, int]:
        return (
            self.worker.speculative_num_steps,
            self.worker.topk,
            self.worker.speculative_num_draft_tokens,
        )

    def register(
        self, state: SpecRuntimeState, config: tuple[int, int, int] | None = None
    ) -> None:
        """Register a pre-built runtime state (e.g. the initial captured state)."""
        self._states[config if config is not None else self._key(state)] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register a runtime state for every distinct table config."""
        self.table.set_cuda_graph_bs(cuda_graph_bs)

        for (steps, topk, draft_tokens) in self.table.configs:
            if (steps, topk, draft_tokens) in self._states:
                continue
            pruned_bs = self.table.cuda_graph_bs_for_config((steps, topk, draft_tokens))
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=draft_tokens,
                topk=topk,
                cuda_graph_bs=pruned_bs,
            )
            self._states[(steps, topk, draft_tokens)] = state

        # Start on the config matching the worker's initial settings.
        if self._worker_config() in self._states:
            self._activate(self._worker_config())

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.table.get_config_for_batch(batch_size)
        if target != self._worker_config():
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Static router does no runtime learning; verify results are ignored."""
        return None

    def _activate(self, config: tuple[int, int, int]) -> None:
        state = self._states.get(config)
        if state is None:
            raise ValueError(f"Missing router runtime state for config {config}")
        self.worker.apply_runtime_state(state)
