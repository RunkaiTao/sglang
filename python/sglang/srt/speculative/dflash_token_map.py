"""Private reduced-vocabulary projection for standard DFLASH draft sampling."""

import os
from typing import Optional

import torch
from torch import nn


def validate_dflash_token_map(token_ids, vocab_size: Optional[int] = None) -> torch.Tensor:
    """Validate on CPU, preserving the file's compact-ID ordering."""
    ids = torch.as_tensor(token_ids, device="cpu")
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("DFLASH token map must be a non-empty 1D table.")
    if ids.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("DFLASH token map must contain integer token IDs.")
    ids = ids.to(torch.int64)
    if ids.min().item() < 0:
        raise ValueError("DFLASH token map IDs must be non-negative.")
    if vocab_size is not None and ids.max().item() >= vocab_size:
        raise ValueError(f"DFLASH token map IDs must lie in [0, {vocab_size}).")
    if ids.unique().numel() != ids.numel():
        raise ValueError("DFLASH token map must not contain duplicate IDs.")
    return ids


def load_dflash_token_map(token_map_path: str) -> torch.Tensor:
    """Load and validate a map without changing the legacy MTP/EAGLE loader."""
    if not os.path.exists(token_map_path):
        from sglang.srt.environ import envs

        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)
        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path
        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                from huggingface_hub import snapshot_download as download_func
            cache_dir = download_func(repo_id, ignore_patterns=["*.bin", "*.safetensors"])
        token_map_path = os.path.join(cache_dir, file_name)
    return validate_dflash_token_map(
        torch.load(token_map_path, weights_only=True, map_location="cpu")
    )


class DFlashTokenMapHead(nn.Module):
    """Select target-head rows once, using file positions as compact token IDs.

    Each rank stores its selected original-vocabulary rows. Greedy sampling
    exchanges only maxima and compact IDs, then returns original token IDs.
    Ties favor the first file position regardless of TP rank or original ID.
    This helper belongs to the DFLASH worker, not to a model's logits processor.
    """

    def __init__(
        self,
        lm_head,
        token_ids,
        *,
        vocab_size: int,
        tp_size: int = 1,
        tp_group=None,
        use_fp32: bool = False,
    ):
        super().__init__()
        weight = getattr(lm_head, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                "--speculative-token-map requires a dense FP16/BF16/FP32 "
                "target lm_head."
            )
        ids = validate_dflash_token_map(token_ids, vocab_size)
        self.vocab_size = int(ids.numel())
        self.use_fp32 = use_fp32
        self.tp_group = tp_group
        self.tp_size = tp_group.world_size if tp_group is not None else int(tp_size)
        if int(getattr(lm_head, "tp_size", self.tp_size)) != self.tp_size:
            raise ValueError(
                "DFLASH token map requires matching target-head "
                "and draft TP sizes."
            )
        shard = getattr(lm_head, "shard_indices", None)
        if shard is None:
            if self.tp_size != 1:
                raise ValueError("DFLASH token map requires head shards for TP>1.")
            start, end = 0, int(vocab_size)
        else:
            if int(shard.num_added_elements) != 0:
                raise ValueError(
                    "DFLASH token map does not support added-vocabulary shards."
                )
            start = int(shard.org_vocab_start_index)
            end = start + int(shard.num_org_elements)
        if not 0 <= start <= end <= vocab_size or end - start > weight.shape[0]:
            raise ValueError("DFLASH token map found invalid head shard bounds.")
        if self.tp_size == 1 and (start != 0 or end != vocab_size):
            raise ValueError("TP=1 token map requires the full target head.")

        positions = torch.arange(ids.numel())[(ids >= start) & (ids < end)]
        local_rows = (ids[positions] - start).to(weight.device)
        self.register_buffer("token_ids", ids.to(weight.device))
        self.register_buffer("local_map_indices", positions.to(weight.device))
        # Allocate only selected rows. Never clone or modify the full target head.
        self.weight = nn.Parameter(
            weight.detach().index_select(0, local_rows), requires_grad=False
        )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.weight, self.token_ids, self.local_map_indices)
        )

    def _group(self, tp_group):
        group = self.tp_group if tp_group is None else tp_group
        if group is None or group.world_size != self.tp_size:
            raise ValueError("DFLASH token map TP group does not match the head.")
        return group

    def _project(self, hidden_states):
        if self.use_fp32:
            if hidden_states.is_cuda and hidden_states.dtype == self.weight.dtype:
                if self.weight.dtype in (torch.float16, torch.bfloat16):
                    return torch.mm(hidden_states, self.weight.T, out_dtype=torch.float32)
            return torch.matmul(hidden_states.float(), self.weight.float().T)
        return torch.matmul(hidden_states.to(self.weight.dtype), self.weight.T)

    def greedy(self, hidden_states, tp_group=None):
        """Argmax over selected rows, returning target/tokenizer IDs."""
        n = hidden_states.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.int64, device=hidden_states.device)
        logits = self._project(hidden_states)
        if self.tp_size == 1:
            return self.token_ids[torch.argmax(logits, dim=-1)]
        sentinel = torch.iinfo(torch.int64).max
        if self.local_map_indices.numel():
            values, indices = torch.max(logits, dim=-1)
            compact_ids = self.local_map_indices[indices]
        else:
            values = logits.new_full((n,), float("-inf"))
            compact_ids = torch.full(
                (n,), sentinel, dtype=torch.int64, device=hidden_states.device
            )
        group = self._group(tp_group)
        gathered_values = values.new_empty(self.tp_size * n)
        gathered_ids = compact_ids.new_empty(self.tp_size * n)
        group.all_gather_into_tensor(gathered_values, values)
        group.all_gather_into_tensor(gathered_ids, compact_ids)
        gathered_values = gathered_values.view(self.tp_size, n)
        gathered_ids = gathered_ids.view(self.tp_size, n)
        best_values = gathered_values.amax(dim=0, keepdim=True)
        nan_values = torch.isnan(gathered_values)
        matches = torch.where(
            nan_values.any(dim=0, keepdim=True),
            nan_values,
            gathered_values == best_values,
        )
        first_position = torch.where(matches, gathered_ids, sentinel).amin(dim=0)
        return self.token_ids[first_position]
