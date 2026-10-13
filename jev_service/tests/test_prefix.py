"""Shared-state trie encoding against independent causal paths.

Runs on CPU with a tiny randomly initialized Qwen3, so no weights are
downloaded. Skipped where torch or transformers is not installed.
"""
import random
import types
import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3Model
except ImportError:
    torch = None


def naive_tries(paths, groups, n_groups):
    nodes = [[] for _ in range(n_groups)]; depths = [[] for _ in range(n_groups)]
    edges = [{} for _ in range(n_groups)]; endpoints = []
    for path, gi in zip(paths, groups):
        parent = -1
        for depth, token in enumerate(path):
            if (parent, token) not in edges[gi]:
                edges[gi][parent, token] = len(nodes[gi]); nodes[gi].append(token); depths[gi].append(depth)
            parent = edges[gi][parent, token]
        endpoints.append(parent)
    return nodes, depths, endpoints


@unittest.skipIf(torch is None, 'requires torch and transformers')
class PrefixTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=101, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                             max_position_embeddings=512)
        config._attn_implementation = 'sdpa'
        cls.backbone = Qwen3Model(config).eval()

    def engine(self, **limits):
        return types.SimpleNamespace(
            torch=torch, device='cpu', path_batch=3, tokenizer=types.SimpleNamespace(pad_token_id=0),
            model=types.SimpleNamespace(path_encoder=types.SimpleNamespace(backbone=self.backbone)), **limits)

    STATES = [(40, [[3, 5], [2, 2, 6, 4], [7, 1, 3]]), (25, [[4, 9]])]

    def request(self, states=STATES):
        """Uneven states and questions, and a candidate that is a prefix of a sibling."""
        rng = random.Random(1)
        tokens = lambda n: [rng.randrange(1, 101) for _ in range(n)]
        paths, locations, rows, question_states = [], [], [], []
        for si, (state_length, candidates) in enumerate(states):
            state = tokens(state_length)
            for counts in candidates:
                qi = len(rows); question = state + tokens(6)
                options = [tokens(n) for n in counts]
                if len(options) > 2:
                    options[1] = options[0] + options[1]  # endpoint inside a sibling's branch
                rows.append({'candidates': options}); question_states.append(si)
                for ci, option in enumerate(options):
                    paths.append(question + option); locations.append((qi, ci))
        return paths, locations, rows, question_states

    def encode(self, mode, engine=None, states=STATES):
        from jev_service.prefix import encode
        paths, locations, rows, question_states = self.request(states)
        with torch.inference_mode():
            return encode(engine or self.engine(), paths, locations, rows, mode, question_states)

    def test_tries_match_token_by_token_insertion(self):
        from agentjev.data import build_tries
        rng = random.Random(0)
        for _ in range(50):
            base = [rng.randrange(4) for _ in range(rng.randrange(1, 12))]
            paths = [base[:rng.randrange(len(base) + 1)] + [rng.randrange(4) for _ in range(rng.randrange(1, 6))]
                     for _ in range(rng.randrange(1, 9))]
            groups = [rng.randrange(3) for _ in paths]
            nodes, depths, _, _, endpoints = build_tries(paths, groups, 3)
            self.assertEqual((nodes, depths, endpoints), naive_tries(paths, groups, 3))

    def test_state_trie_matches_independent_paths(self):
        reference, mask, path_usage = self.encode('path')
        for mode in ('auto', 'question', 'shared'):
            with self.subTest(mode=mode):
                vectors, other_mask, usage = self.encode(mode)
                self.assertTrue(torch.equal(mask, other_mask))
                self.assertLess((vectors - reference)[mask].abs().max().item(), 1e-4)
        _, _, usage = self.encode('auto')
        self.assertEqual(usage['shared_prefix_states'], 2)
        self.assertEqual(usage['shared_prefix_questions'], 4)
        self.assertLess(usage['backbone_input_tokens'], path_usage['backbone_input_tokens'] / 2)

    def test_flat_forest_and_split_chunks_match_independent_paths(self):
        # A large and a tiny state share one flat row; a tight token budget
        # instead puts each state in its own forward pass.
        states = [(40, [[3, 5], [2, 2, 6, 4]]), (2, [[1, 2]])]
        reference, mask, _ = self.encode('path', states=states)
        for limits in ({}, {'tree_max_tokens': 80}):
            with self.subTest(**limits):
                vectors, _, usage = self.encode('auto', self.engine(**limits), states=states)
                self.assertEqual(usage['shared_prefix_states'], 2)
                self.assertLess((vectors - reference)[mask].abs().max().item(), 1e-4)

    def test_over_budget_state_falls_back(self):
        reference, mask, _ = self.encode('path')
        # The 40-token state's trie no longer fits; the 25-token one does.
        vectors, _, usage = self.encode('auto', self.engine(tree_max_tokens=60))
        self.assertEqual(usage['shared_prefix_states'], 1)
        self.assertLess((vectors - reference)[mask].abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
