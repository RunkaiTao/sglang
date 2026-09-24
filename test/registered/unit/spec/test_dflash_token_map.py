import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_token_map import (
    DFlashTokenMapHead,
    load_dflash_token_map,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _head(weight, start=None, count=None):
    head = SimpleNamespace(weight=weight)
    if start is not None:
        head.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            num_org_elements=count,
            num_added_elements=0,
        )
    return head


class _FakeTpGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.recording = True
        self.inputs = {}
        self.rank = 0
        self.call = 0

    def all_gather_into_tensor(self, output, input_):
        if self.recording:
            self.inputs[self.rank, self.call] = input_.clone()
            output.zero_()
        else:
            output.copy_(
                torch.cat(
                    [self.inputs[r, self.call] for r in range(self.world_size)]
                )
            )
        self.call += 1


class TestDFlashTokenMapHead(unittest.TestCase):
    def test_loader_preserves_order_and_rejects_invalid_ids_before_cast(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.pt"
            for value in ([10, 1, 4], torch.tensor([10, 1, 4], dtype=torch.int32)):
                torch.save(value, path)
                ids = load_dflash_token_map(str(path))
                self.assertEqual(ids.dtype, torch.int64)
                self.assertEqual(ids.device.type, "cpu")
                self.assertEqual(ids.tolist(), [10, 1, 4])
            for value in ([1.9], [True], [1, 1], [[1, 2]], [], [-1]):
                torch.save(value, path)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_dflash_token_map(str(path))

    def test_projects_only_selected_rows_and_returns_original_ids(self):
        weight = torch.tensor(
            [[100.0, 100.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        original = weight.clone()
        head = DFlashTokenMapHead(_head(weight), [3, 2, 1], vocab_size=4)
        hidden = torch.tensor([[2.0, 1.0], [0.0, 3.0], [-2.0, 0.0]])
        torch.testing.assert_close(head.greedy(hidden), torch.tensor([1, 2, 3]))
        self.assertEqual(head.weight.shape, (3, 2))
        self.assertNotEqual(head.weight.data_ptr(), weight.data_ptr())
        torch.testing.assert_close(weight, original)
        self.assertEqual(head.nbytes, 3 * 2 * 4 + 2 * 3 * 8)

    def test_file_order_breaks_ties_and_excludes_padding(self):
        weight = torch.zeros(10, 4)
        weight[2] = 1
        weight[6] = 1
        weight[8:] = 100  # Padded rows must never participate.
        hidden = torch.ones(3, 4)
        token_ids = torch.tensor(list(reversed(range(8))))
        head = DFlashTokenMapHead(_head(weight, start=0, count=8), token_ids, vocab_size=8)
        expected = token_ids[torch.argmax(hidden @ weight[token_ids].T, dim=-1)]
        self.assertEqual(expected[0].item(), 6)
        torch.testing.assert_close(head.greedy(hidden), expected)

    def test_bf16_head_accepts_fp32_hidden(self):
        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        head = DFlashTokenMapHead(_head(weight), [0, 1], vocab_size=2)
        torch.testing.assert_close(
            head.greedy(torch.tensor([[0.0, 1.0]])), torch.tensor([1])
        )
        self.assertEqual(head.greedy(torch.empty(0, 2)).shape, (0,))

    def test_fp32_projection_preserves_a_difference_rounded_away_in_bf16(self):
        weight = torch.tensor([[1.0, 0.0], [1.0, 0.002]], dtype=torch.bfloat16)
        hidden = torch.tensor([[1.0, 0.7]])
        rounded = DFlashTokenMapHead(_head(weight), [0, 1], vocab_size=2)
        fp32 = DFlashTokenMapHead(
            _head(weight), [0, 1], vocab_size=2, use_fp32=True
        )
        self.assertEqual(rounded.greedy(hidden).item(), 0)
        self.assertEqual(fp32.greedy(hidden).item(), 1)
        torch.testing.assert_close(fp32._project(hidden), hidden @ weight.float().T)

    def test_invalid_maps_and_unsupported_heads_fail(self):
        for ids in ([], [[1, 2]], [-1], [8], [1, 1], [1.5], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                DFlashTokenMapHead(_head(torch.zeros(8, 4)), ids, vocab_size=8)
        with self.assertRaisesRegex(ValueError, "dense"):
            DFlashTokenMapHead(
                _head(torch.zeros(8, 4, dtype=torch.int8)), [1], vocab_size=8
            )
        with self.assertRaisesRegex(ValueError, "shards"):
            DFlashTokenMapHead(
                _head(torch.zeros(4, 4)), [1], vocab_size=8, tp_size=2
            )
        with self.assertRaisesRegex(ValueError, "bounds"):
            DFlashTokenMapHead(_head(torch.zeros(4, 4)), [1], vocab_size=8)

    def test_incompatible_tp_and_added_vocab_fail(self):
        source = _head(torch.zeros(4, 4), start=0, count=4)
        source.tp_size = 2
        with self.assertRaisesRegex(ValueError, "matching"):
            DFlashTokenMapHead(source, [1], vocab_size=8)
        head = DFlashTokenMapHead(source, [1], vocab_size=8, tp_size=2)
        with self.assertRaisesRegex(ValueError, "group"):
            head.greedy(torch.ones(1, 4), SimpleNamespace(world_size=3))
        with self.assertRaisesRegex(ValueError, "full target"):
            DFlashTokenMapHead(
                _head(torch.zeros(4, 4), start=0, count=4), [1], vocab_size=8
            )
        source.shard_indices.num_added_elements = 1
        with self.assertRaisesRegex(ValueError, "added-vocabulary"):
            DFlashTokenMapHead(source, [1], vocab_size=8, tp_size=2)

    def test_tp_matches_restricted_full_head_with_empty_rank_and_ties(self):
        for scenario in ("random", "tied", "all_negative_inf", "nan"):
            with self.subTest(scenario=scenario):
                generator = torch.Generator().manual_seed(17)
                weight = torch.randn(24, 8, generator=generator)
                hidden = torch.randn(7, 8, generator=generator)
                if scenario == "tied":
                    weight.zero_()
                    weight[2] = 1
                    weight[15] = 1
                    hidden.abs_()
                elif scenario == "all_negative_inf":
                    weight.fill_(float("-inf"))
                    hidden.fill_(1)
                elif scenario == "nan":
                    weight[2] = float("nan")
                    weight[15] = float("nan")
                ids = [15, 2, 5]
                group = _FakeTpGroup(3)
                heads = [
                    DFlashTokenMapHead(
                        _head(weight[r * 8 : (r + 1) * 8], start=r * 8, count=8),
                        ids,
                        vocab_size=24,
                        tp_size=3,
                    )
                    for r in range(3)
                ]
                self.assertEqual(heads[2].weight.shape[0], 0)
                map_ids = torch.tensor(ids)
                expected = map_ids[
                    (hidden @ weight[map_ids].T).argmax(dim=-1)
                ]
                if scenario != "random":
                    self.assertEqual(expected[0].item(), 15)
                for recording in (True, False):
                    group.recording = recording
                    for rank, head in enumerate(heads):
                        group.rank, group.call = rank, 0
                        actual = head.greedy(hidden, group)
                        if not recording:
                            torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Graph requires a GPU")
    def test_cuda_graph_replay_uses_current_hidden_states(self):
        weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
        head = DFlashTokenMapHead(_head(weight), [7, 13, 31], vocab_size=64)
        hidden = torch.randn(6, 32, device="cuda", dtype=torch.bfloat16)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                head.greedy(hidden)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = head.greedy(hidden)
        hidden.normal_()
        graph.replay()
        torch.testing.assert_close(output, head.greedy(hidden))


class TestDFlashTokenMapWorker(unittest.TestCase):
    def _worker(self):
        from sglang.srt.speculative import dflash_worker_v2 as worker_mod

        worker = worker_mod.DFlashWorkerV2.__new__(worker_mod.DFlashWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=_head(
                        torch.arange(48, dtype=torch.float32).view(12, 4)
                    )
                ),
                model_config=SimpleNamespace(vocab_size=12),
            )
        )
        worker.model_runner = worker._target_worker.model_runner
        worker._draft_worker = SimpleNamespace(preloaded_weights_bytes=100)
        worker.draft_model_runner = SimpleNamespace(tp_group=None)
        worker.draft_tp_context = lambda _: nullcontext()
        worker.selector = None
        worker._is_domino = False
        worker._draft_token_map_head = None
        worker.block_size = 4
        worker.ps = SimpleNamespace(tp_rank=0)
        return worker_mod, worker

    def test_loaded_map_is_used_by_eager_and_graph_sampler(self):
        worker_mod, worker = self._worker()
        group = SimpleNamespace(world_size=1)
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[2]))
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "map.pt")
            torch.save([7, 2, 9], path)
            with (
                patch.object(
                    worker_mod,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_token_map=path),
                ),
                patch.object(worker_mod, "get_tp_group", return_value=group),
                patch.object(worker_mod, "get_exec", return_value=execution),
            ):
                worker._init_draft_token_map()
                self.assertEqual(
                    worker.preloaded_weights_bytes,
                    100 + worker._draft_token_map_head.nbytes,
                )
                full_weight = worker.model_runner.model.lm_head.weight
                self.assertEqual(full_weight.shape, (12, 4))
                sampler = worker._maybe_build_draft_sampler()
                hidden = torch.ones(8, 4)
                hidden[4:] *= -1
                sampler(hidden)
                proposal_hidden = hidden.view(2, 4, 4)[:, 1:].reshape(-1, 4)
                eager = worker._greedy_sample_from_vocab_parallel_head(
                    hidden_states=proposal_hidden,
                    lm_head=worker.model_runner.model.lm_head,
                )
                expected = torch.tensor([9, 9, 9, 2, 2, 2])
                torch.testing.assert_close(eager, expected)
                torch.testing.assert_close(sampler.out[:6], expected)

    def test_no_map_preserves_existing_full_vocab_sampler(self):
        worker_mod, worker = self._worker()
        with patch.object(
            worker_mod,
            "get_spec",
            return_value=SimpleNamespace(speculative_token_map=None),
        ):
            worker._init_draft_token_map()
        self.assertIsNone(worker._draft_token_map_head)
        self.assertEqual(worker.preloaded_weights_bytes, 100)
        weight = worker.model_runner.model.lm_head.weight
        sampler = worker_mod._DflashDraftSampler(
            weight=weight, block_size=4, num_org=12, org_vocab_start=0, max_bs=1
        )
        sampler(torch.ones(4, 4))
        torch.testing.assert_close(sampler.out, torch.tensor([11, 11, 11]))

    def test_worker_inherits_target_projection_precision(self):
        worker_mod, worker = self._worker()
        worker.model_runner.model.logits_processor = SimpleNamespace(
            use_fp32_lm_head=True
        )
        with (
            patch.object(
                worker_mod,
                "get_spec",
                return_value=SimpleNamespace(speculative_token_map="map.pt"),
            ),
            patch.object(
                worker_mod, "load_dflash_token_map", return_value=torch.tensor([7, 2, 9])
            ),
            patch.object(
                worker_mod, "get_tp_group", return_value=SimpleNamespace(world_size=1)
            ),
        ):
            worker._init_draft_token_map()
        self.assertTrue(worker._draft_token_map_head.use_fp32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Graph requires a GPU")
    def test_cuda_graph_sampler_matches_eager_and_restricted_reference(self):
        worker_mod, worker = self._worker()
        for use_fp32 in (False, True):
            with self.subTest(use_fp32=use_fp32):
                generator = torch.Generator(device="cuda").manual_seed(42)
                weight = torch.randn(
                    512, 32, generator=generator, device="cuda", dtype=torch.bfloat16
                )
                weight[401] = weight[7]
                ids = torch.tensor([401, 7, 129], device="cuda")
                head = DFlashTokenMapHead(
                    _head(weight), ids, vocab_size=512, use_fp32=use_fp32
                )
                worker._draft_token_map_head = head
                sampler = worker_mod._DflashDraftSampler(
                    weight=head.weight, block_size=4, num_org=3,
                    org_vocab_start=0, max_bs=2, token_map_head=head,
                )
                hidden = torch.randn(
                    8, 32, generator=generator, device="cuda", dtype=torch.bfloat16
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        sampler(hidden)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    sampler(hidden)
                output_ptr = sampler.out.data_ptr()
                for _ in range(3):
                    hidden.normal_(generator=generator)
                    graph.replay()
                    selected = hidden.view(2, 4, 32)[:, 1:].reshape(-1, 32)
                    selected_weight = weight[ids]
                    reference_logits = (
                        torch.mm(selected, selected_weight.T, out_dtype=torch.float32)
                        if use_fp32 else selected @ selected_weight.T
                    )
                    expected = ids[reference_logits.argmax(-1)]
                    with patch.object(
                        worker_mod, "get_tp_group",
                        return_value=SimpleNamespace(world_size=1),
                    ):
                        eager = worker._greedy_sample_from_vocab_parallel_head(
                            hidden_states=selected, lm_head=_head(weight)
                        )
                    torch.testing.assert_close(sampler.out[:6], expected)
                    torch.testing.assert_close(eager, expected)
                    self.assertEqual(sampler.out.data_ptr(), output_ptr)

    def test_map_with_selector_or_domino_is_rejected(self):
        worker_mod, worker = self._worker()
        with patch.object(
            worker_mod,
            "get_spec",
            return_value=SimpleNamespace(speculative_token_map="map.pt"),
        ):
            for selector, domino in ((object(), False), (None, True)):
                worker.selector, worker._is_domino = selector, domino
                with self.assertRaisesRegex(ValueError, "standard"):
                    worker._init_draft_token_map()

    def test_target_can_commit_a_bonus_outside_the_draft_map(self):
        worker_mod, worker = self._worker()
        worker._selector_sample = None
        worker._use_triton_accept_bonus = False
        worker._tp_sync = SimpleNamespace(sync=lambda *args: None)
        candidates = torch.tensor([[0, 2, 7, 9]])
        logits = torch.zeros(4, 12)
        logits[torch.arange(4), torch.tensor([2, 11, 5, 9])] = 10
        accept, commit, bonus, out, _, _ = worker._accept_block(
            candidates=candidates,
            next_token_logits=logits,
            sampling_info=SimpleNamespace(is_all_greedy=True),
            draft_input=None,
            prefix_lens=torch.tensor([10]),
            bs=1,
        )
        self.assertEqual(accept.item(), 1)
        self.assertEqual(commit.item(), 2)
        self.assertEqual(bonus.item(), 11)
        torch.testing.assert_close(out[0, :2], torch.tensor([2, 11]))


if __name__ == "__main__":
    unittest.main()
