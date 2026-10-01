"""Fetch the pinned benchmark v1 weights, verify them, and wrap them for the server.

The quickstart used to describe this step in prose with an unpinned download.
That is not reproducible today: the model repository's default branch serves a
later coding-completion checkpoint (revision 0e2593e6, 2,393,718,620 bytes, all
F32), while the benchmark v1 release is revision 7d433994 (1,196,881,242 bytes,
all BF16). File size and dtype follow the revision, so this script pins the
revision and refuses anything that does not match.

It also fetches the Qwen3 base model into a local directory. The server loads its
tokenizer with ``local_files_only=True``, so a bare repository id only works when
the reader happens to have that model cached already; the printed command passes
the downloaded directory instead.

torch, huggingface_hub and safetensors are imported inside ``main()``, so this
module - and the test that guards its verification logic - import on a checkout
that has not installed them.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

MODEL_REPO = 'aimeigaoshou/agent-jev'
BASE_MODEL = 'Qwen/Qwen3-0.6B'
V1_REVISION = '7d433994fbde17a3f0993c2f2b02fe8ca1370db1'
V1_FILES = {
    'model.safetensors': {'bytes': 1196881242,
                          'sha256': '8166e46dc6019ae13f0d8fc97d603cdbcc2d20eb1882c7350c8deb9fc6bae215'},
    'temperatures.json': {'bytes': 389,
                          'sha256': '5e5032896d77c72feb86b4725abe418488b8a1c7476014c706bd8215eb43c86c'},
}
V1_DTYPE = 'BF16'
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=Path('agentjev_v1.pt'),
                        help='torch checkpoint the server loads')
    parser.add_argument('--temperatures-output', type=Path,
                        help='where to write temperatures.json (default: beside --output)')
    parser.add_argument('--base-model-dir', type=Path, default=Path('models/Qwen3-0.6B'),
                        help='local directory for the Qwen3 base model used for the skeleton')
    parser.add_argument('--skip-base-model', action='store_true',
                        help='do not fetch the base model; pass your own --model-path instead')
    args = parser.parse_args(argv)

    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    import torch

    release = {}
    for name, expected in sorted(V1_FILES.items()):
        path = Path(hf_hub_download(MODEL_REPO, name, revision=V1_REVISION))
        digest = verify_file(path, expected)
        release[name] = path
        print(f'{name}: {expected["bytes"]} bytes, sha256 {digest[:16]}... ok')

    counts = safetensors_dtypes(release['model.safetensors'])
    if set(counts) != {V1_DTYPE}:
        raise ValueError(f'revision {V1_REVISION} should be all {V1_DTYPE}; header reports {counts}')

    state_dict = load_file(str(release['model.safetensors']))
    loaded = {str(tensor.dtype).replace('torch.', '') for tensor in state_dict.values()}
    if loaded != {SAFETENSORS_TO_TORCH[V1_DTYPE]}:
        raise ValueError(f'loaded tensors are {sorted(loaded)}, expected {SAFETENSORS_TO_TORCH[V1_DTYPE]}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state_dict}, args.output)
    temperatures = args.temperatures_output or args.output.parent / 'temperatures.json'
    temperatures.write_bytes(release['temperatures.json'].read_bytes())

    if not args.skip_base_model:
        snapshot_download(BASE_MODEL, local_dir=str(args.base_model_dir))

    print(json.dumps({'revision': V1_REVISION, 'tensors': len(state_dict), 'dtype': V1_DTYPE,
                      'checkpoint': str(args.output), 'checkpoint_sha256': sha256_of(args.output),
                      'temperatures': str(temperatures),
                      'base_model_dir': None if args.skip_base_model else str(args.base_model_dir)},
                     indent=2))
    print('\nStart the service with:\n')
    print(f'python -m jev_service.server --checkpoint {args.output} '
          f'--model-path {"<your model path>" if args.skip_base_model else args.base_model_dir} '
          f'--temperatures {temperatures} --port 8149')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
