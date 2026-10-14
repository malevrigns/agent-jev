"""Fetch the pinned benchmark v1 release and verify every file the server reads.

The model repository's default branch serves a later coding-completion
checkpoint (revision 0e2593e6, 2,393,718,620 bytes, all F32), while the
benchmark v1 release is revision 7d433994 (1,196,881,242 bytes, all BF16).
File size and dtype follow the revision, so this script pins the revision and
refuses anything that does not match.

That revision also carries the Qwen3-0.6B-Base config and tokenizer the model
was trained with, so nothing else is downloaded: the server builds the Qwen3
skeleton from config.json and loads model.safetensors over it. Every file is
pinned by size and SHA-256 and lands in one directory, which is passed to the
server as both the checkpoint folder and --model-path.

``--variant coding-gates`` fetches the coding-gates candidate instead: v1
continued on coding_gates/ (branch ``coding-gates``, made from the v1 commit).
It keeps v1's config and tokenizer and changes the weights and temperatures.

huggingface_hub is imported inside ``main()``, so this module - and the test
that guards its verification logic - import on a checkout that has not
installed it.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

MODEL_REPO = 'aimeigaoshou/agent-jev'
V1_REVISION = '7d433994fbde17a3f0993c2f2b02fe8ca1370db1'
V1_FILES = {
    'model.safetensors': {'bytes': 1196881242,
                          'sha256': '8166e46dc6019ae13f0d8fc97d603cdbcc2d20eb1882c7350c8deb9fc6bae215'},
    'temperatures.json': {'bytes': 389,
                          'sha256': '5e5032896d77c72feb86b4725abe418488b8a1c7476014c706bd8215eb43c86c'},
    'config.json': {'bytes': 727,
                    'sha256': '504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59'},
    'tokenizer.json': {'bytes': 7031645,
                       'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'},
    'tokenizer_config.json': {'bytes': 9678,
                              'sha256': '3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5'},
    'vocab.json': {'bytes': 2776833,
                   'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'},
    'merges.txt': {'bytes': 1671853,
                   'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'},
}
V1_DTYPE = 'BF16'
V1_TENSORS = 343
GATES_REVISION = '359aa94b1af47ba4a2b2f969e99f40516cc3054e'
GATES_FILES = dict(V1_FILES, **{
    'model.safetensors': {'bytes': 1196881242,
                          'sha256': 'd8e39f94dc949f459b5b5ce254311a2ef99c010b0036b847dc670bebf3a8f2d8'},
    'temperatures.json': {'bytes': 403,
                          'sha256': '7716193b27bb1276155e9ab467fe2897028e1a9c709be18a159242416418b69f'},
})
# variant: (revision, files, default directory); both hold 343 BF16 tensors
RELEASES = {'v1': (V1_REVISION, V1_FILES, Path('models/agentjev-v1')),
            'coding-gates': (GATES_REVISION, GATES_FILES, Path('models/agentjev-coding-gates'))}
SAFETENSORS_TO_TORCH = {'BF16': 'bfloat16', 'F16': 'float16', 'F32': 'float32'}


def sha256_of(path, block_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    """Return the digest, or raise if the file is not the release artifact."""
    path = Path(path)
    size = path.stat().st_size
    if size != expected['bytes']:
        raise ValueError(f'{path.name}: {size} bytes, expected {expected["bytes"]}')
    digest = sha256_of(path)
    if digest != expected['sha256']:
        raise ValueError(f'{path.name}: sha256 {digest}, expected {expected["sha256"]}')
    return digest


def safetensors_dtypes(path):
    """Count tensor dtypes from the header only; no tensor data is read."""
    with open(path, 'rb') as stream:
        header_bytes = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_bytes).decode('utf-8'))
    counts = {}
    for name, entry in header.items():
        if name != '__metadata__':
            counts[entry['dtype']] = counts.get(entry['dtype'], 0) + 1
    return counts


def serve_command(model_dir, port=8149):
    model_dir = Path(model_dir)
    return (f'python -m jev_service.server --checkpoint {model_dir / "model.safetensors"} '
            f'--model-path {model_dir} --temperatures {model_dir / "temperatures.json"} --port {port}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--variant', choices=sorted(RELEASES), default='v1',
                        help='v1 (default) or the coding-gates candidate')
    parser.add_argument('--model-dir', type=Path,
                        help='directory that receives the weights, temperatures, config and tokenizer '
                             '(default: models/agentjev-v1, or models/agentjev-coding-gates)')
    args = parser.parse_args(argv)
    revision, files, default_dir = RELEASES[args.variant]
    model_dir = args.model_dir or default_dir

    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        path = Path(hf_hub_download(MODEL_REPO, name, revision=revision, local_dir=str(model_dir)))
        digest = verify_file(path, expected)
        print(f'{name}: {expected["bytes"]} bytes, sha256 {digest[:16]}... ok')

    counts = safetensors_dtypes(model_dir / 'model.safetensors')
    if counts != {V1_DTYPE: V1_TENSORS}:
        raise ValueError(f'revision {revision} should hold {V1_TENSORS} {V1_DTYPE} tensors; header reports {counts}')

    print(json.dumps({'variant': args.variant, 'revision': revision, 'tensors': V1_TENSORS, 'dtype': V1_DTYPE,
                      'model_dir': str(model_dir), 'files': sorted(files)}, indent=2))
    print('\nStart the service with:\n')
    print(serve_command(model_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
