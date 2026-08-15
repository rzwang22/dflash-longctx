from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # Mac base environment deliberately omits model frameworks.
    torch = None

from dflash.phase2c.core import (
    build_online_tree_tensors,
    build_tree_attention_mask,
    compact_dynamic_cache,
    greedy_tree_traversal_tensors,
    layerwise_beam_tree,
    tensor_tree_metadata,
    tree_position_tensors,
)


@unittest.skipIf(torch is None, "optional CPU Torch tensor tests")
class Phase2CTensorTests(unittest.TestCase):
    def _candidates(self):
        token_ids = torch.tensor(
            [[100 * (depth + 1) + rank for rank in (1, 2, 3)] for depth in range(15)],
            dtype=torch.long,
        )
        scores = torch.tensor(
            [
                [-1.0 - depth / 1000, -2.0 - depth / 1000, -3.0 - depth / 1000]
                for depth in range(15)
            ],
            dtype=torch.float32,
        )
        return {"token_ids": token_ids, "log_probabilities": scores, "top_k": 3}

    def _tree(self, *, budget=32, width=4):
        return build_online_tree_tensors(
            torch,
            candidates=self._candidates(),
            root_token_id=torch.tensor(42),
            policy="layerwise_beam",
            node_budget=budget,
            beam_width=width,
        )

    def test_tensor_policy_matches_phase2b1_cpu_policy(self) -> None:
        candidates = self._candidates()
        positions = []
        for depth in range(15):
            positions.append(
                {
                    "candidates": [
                        {
                            "rank": rank + 1,
                            "token_id": int(candidates["token_ids"][depth, rank]),
                            "log_probability": float(
                                candidates["log_probabilities"][depth, rank]
                            ),
                        }
                        for rank in range(3)
                    ]
                }
            )
        expected = layerwise_beam_tree(positions, budget=32, beam_width=4)
        metadata = tensor_tree_metadata(self._tree())
        actual = {tuple(node["draft_token_path"]) for node in metadata["nodes"]}
        self.assertEqual(actual, expected)

    def test_tensor_mask_enforces_tree_visibility(self) -> None:
        tree = self._tree(budget=16, width=4)
        mask = build_tree_attention_mask(
            torch, tree=tree, prefix_length=3, dtype=torch.float32
        )[0, 0]
        metadata = tensor_tree_metadata(tree)
        paths = {
            tuple(node["draft_token_path"]): node["node_index"]
            for node in metadata["nodes"]
        }
        depth_one = [path for path in paths if len(path) == 1]
        query = paths[depth_one[0]]
        sibling = paths[depth_one[1]]
        self.assertTrue(torch.all(mask[query, :3] == 0))
        self.assertEqual(float(mask[query, 3]), 0.0)
        self.assertEqual(float(mask[query, 3 + query]), 0.0)
        self.assertLess(float(mask[query, 3 + sibling]), 0.0)

    def test_tensor_positions_separate_semantic_and_storage(self) -> None:
        tree = self._tree(budget=16, width=4)
        semantic, storage = tree_position_tensors(
            torch, tree=tree, context_length_before=100
        )
        self.assertTrue(torch.equal(semantic[0], 100 + tree["depths"]))
        self.assertTrue(
            torch.equal(
                storage, torch.arange(100, 100 + tree["actual_total_node_count"])
            )
        )

    def test_tensor_greedy_traversal(self) -> None:
        tree = self._tree(budget=16, width=4)
        vocabulary = 2000
        logits = torch.full((1, tree["actual_total_node_count"], vocabulary), -100.0)
        children = {}
        for index, parent in enumerate(tree["parent_indices"].tolist()):
            if parent >= 0:
                children.setdefault(parent, []).append(index)
        first = children[0][0]
        logits[0, 0, int(tree["token_ids"][first])] = 10.0
        logits[0, first, 1999] = 10.0
        result = greedy_tree_traversal_tensors(torch, tree=tree, target_logits=logits)
        self.assertEqual(result["accepted_draft_token_count"], 1)
        self.assertEqual(int(result["next_root_token_id"]), 1999)

    def test_tensor_multilayer_path_aware_cache_compaction(self) -> None:
        class Layer:
            def __init__(self, offset):
                values = torch.arange(offset, offset + 8, dtype=torch.float32)
                self.keys = values.reshape(1, 1, 8, 1)
                self.values = (values + 100).reshape(1, 1, 8, 1)

            def get_seq_length(self):
                return self.keys.shape[-2]

        class Cache:
            def __init__(self):
                self.layers = [Layer(0), Layer(10)]

        cache = Cache()
        result = compact_dynamic_cache(
            torch,
            cache=cache,
            prefix_length=2,
            actual_tree_node_count=6,
            accepted_node_indices=torch.tensor([2, 5]),
        )
        self.assertEqual(result["compacted_layer_lengths"], [5, 5])
        self.assertEqual(cache.layers[0].keys.flatten().tolist(), [0, 1, 2, 4, 7])
        self.assertNotIn(3, cache.layers[0].keys.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
