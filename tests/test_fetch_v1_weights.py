"""Guards for the pinned-release fetch helper. No network, and no torch."""
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from fetch_v1_weights import (SAFETENSORS_TO_TORCH, V1_DTYPE, V1_FILES, V1_REVISION,
                              safetensors_dtypes, sha256_of, verify_file)


def write_safetensors(path, dtypes):
    header = {f'tensor_{index}': {'dtype': dtype, 'shape': [2], 'data_offsets': [0, 2]}
              for index, dtype in enumerate(dtypes)}
    blob = json.dumps(header).encode('utf-8')
    # Header only: no tensor payload follows, so a reader that touches tensors would fail.
    path.write_bytes(struct.pack('<Q', len(blob)) + blob)


class TemporaryDirectoryTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()


class VerifyFileTests(TemporaryDirectoryTest):
    def test_matching_file_returns_its_digest(self):
        target = self.tmp / 'temperatures.json'
        payload = b'{"boolean": {"temperature": 1.0}}'
        target.write_bytes(payload)
        expected = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        self.assertEqual(verify_file(target, expected), expected['sha256'])
        self.assertEqual(sha256_of(target), expected['sha256'])

    def test_wrong_size_is_rejected(self):
        target = self.tmp / 'temperatures.json'
        target.write_bytes(b'short')
        with self.assertRaises(ValueError) as caught:
            verify_file(target, {'bytes': 99, 'sha256': hashlib.sha256(b'short').hexdigest()})
        self.assertIn('temperatures.json', str(caught.exception))
        self.assertIn('expected 99', str(caught.exception))

    def test_wrong_digest_is_rejected(self):
        target = self.tmp / 'model.safetensors'
        target.write_bytes(b'not the release artifact')
        with self.assertRaises(ValueError) as caught:
            verify_file(target, {'bytes': 24, 'sha256': '0' * 64})
        self.assertIn('sha256', str(caught.exception))
        self.assertIn('0' * 64, str(caught.exception))

class HeaderTests(TemporaryDirectoryTest):
    def test_dtypes_come_from_the_header_alone(self):
        target = self.tmp / 'model.safetensors'
        write_safetensors(target, ['BF16', 'BF16', 'BF16'])
        self.assertEqual(safetensors_dtypes(target), {'BF16': 3})
        write_safetensors(target, ['F32', 'BF16'])
        self.assertEqual(safetensors_dtypes(target), {'F32': 1, 'BF16': 1})


class PinnedReleaseTests(unittest.TestCase):
    def test_revision_and_artifacts_are_fully_pinned(self):
        self.assertEqual(len(V1_REVISION), 40)
        int(V1_REVISION, 16)
        self.assertEqual(V1_DTYPE, 'BF16')
        self.assertEqual(SAFETENSORS_TO_TORCH[V1_DTYPE], 'bfloat16')
        self.assertEqual(set(V1_FILES), {'model.safetensors', 'temperatures.json'})
        for name, expected in V1_FILES.items():
            with self.subTest(name=name):
                self.assertIsInstance(expected['bytes'], int)
                self.assertGreater(expected['bytes'], 0)
                self.assertEqual(len(expected['sha256']), 64)
                int(expected['sha256'], 16)


if __name__ == '__main__':
    unittest.main()
