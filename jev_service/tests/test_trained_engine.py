import math
import threading
import unittest

import torch

from agentjev.data import make_collate
from jev_service.engine import DecisionEngine


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(char) + 2 for char in text]}


class FixedModel:
    def __call__(self, batch):
        logits = torch.tensor([[2.0, 0.0]], device=batch['cand_mask'].device)
        return {'logits': logits.expand(batch['cand_mask'].shape[0], -1)}


class TrainedEngineTests(unittest.TestCase):
    def test_checkpoint_collator_batches_and_calibrates(self):
        engine = DecisionEngine.__new__(DecisionEngine)
        engine.torch = torch
        engine.device = 'cpu'
        engine.lock = threading.Lock()
        engine.trained_config = {'batch_states': 1, 'max_len': 128}
        engine.collate = make_collate(Tokenizer(), max_len=128)
        engine.model = FixedModel()
        engine.temperatures = {'boolean': 2.0}
        engine.checkpoint_name = 'test'
        payload = {'requests': [
            {'id': 'first', 'state': 's', 'questions': [
                {'id': 'q1', 'type': 'boolean', 'question': 'yes?'}]},
            {'id': 'second', 'state': 's', 'questions': [
                {'id': 'q2', 'type': 'choice', 'question': 'pick?',
                 'options': {'a': 'A', 'b': 'B'}}]},
        ]}

        result = engine.evaluate(payload)

        self.assertEqual([row['id'] for row in result['results']], ['first', 'second'])
        answers = [row['answers'][0] for row in result['results']]
        self.assertEqual([answer['id'] for answer in answers], ['q1', 'q2'])
        self.assertAlmostEqual(answers[0]['probability'], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(answers[1]['top_probability'], 1 / (1 + math.exp(-2)))
        self.assertEqual(result['usage']['questions'], 2)
        self.assertEqual(result['usage']['candidate_paths'], 4)


if __name__ == '__main__':
    unittest.main()
