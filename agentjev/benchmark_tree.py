"""Compare PathEncoder with the final TreeEncoder on identical weights.

All timings include CPU tree packing (Tree only) and host-to-device transfer.
Training includes zero_grad, forward and backward, but not optimizer steps,
tokenization, data loading or the candidate head. No weights are downloaded.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model, __version__ as transformers_version

from agentjev.encoders import PathEncoder, TreeEncoder
from agentjev.data import batch_to_device, pack_tree_batch


SCENARIOS = {
    "short_prefix": {
        "description": "Two small questions with little prefix reuse",
        "candidates": [4, 4], "prefix_lengths": [16, 16], "suffix_lengths": [12, 12],
    },
    "long_prefix": {
        "description": "One question with a long shared prefix and many candidates",
        "candidates": [64], "prefix_lengths": [192], "suffix_lengths": [8],
    },
    "uneven_questions": {
        "description": "One large question and three smaller questions in the same batch",
        "candidates": [32, 4, 2, 1], "prefix_lengths": [128, 32, 16, 8],
        "suffix_lengths": [8, 8, 8, 8],
    },
}

def make_batch(spec: dict, vocab_size: int, seed: int, scale: int = 1) -> dict:
    """Make independent questions with exact, reproducible prefix sharing."""
    rng = torch.Generator().manual_seed(seed)
    paths, locations = [], []
    if max(spec["candidates"]) >= vocab_size - 1:
        raise ValueError("Vocabulary is too small to create distinct candidate branches")
    for qi, (count, prefix_length, suffix_length) in enumerate(zip(
            spec["candidates"], spec["prefix_lengths"], spec["suffix_lengths"])):
        prefix = torch.randint(1, vocab_size, (prefix_length * scale,), generator=rng).tolist()
        for ci in range(count):
            # The first suffix token differs, so branches never accidentally share.
            suffix = [ci + 1] + torch.randint(
                1, vocab_size, (suffix_length * scale - 1,), generator=rng).tolist()
            paths.append(prefix + suffix)
            locations.append((qi, ci))
    lengths = torch.tensor([len(path) for path in paths])
    ids = torch.zeros(len(paths), int(lengths.max()), dtype=torch.long)
    for row, path in enumerate(paths):
        ids[row, :len(path)] = torch.tensor(path)
    q_index, cand_index = torch.tensor(locations).unbind(1)
    cand_mask = torch.zeros(len(spec["candidates"]), max(spec["candidates"]), dtype=torch.bool)
    cand_mask[q_index, cand_index] = True
    return {
        "input_ids": ids,
        "attention_mask": (torch.arange(ids.size(1)) < lengths[:, None]).long(),
        "cand_end_pos": lengths - 1, "q_index": q_index,
        "cand_index": cand_index, "cand_mask": cand_mask,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(backbone: nn.Module, device: torch.device) -> None:
    backbone.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync(device)


def summarize_times(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
        "p90_ms": ordered[max(0, math.ceil(len(ordered) * 0.9) - 1)], "samples_ms": samples,
    }


def measure(encoder: nn.Module, batch: dict, mode: str,
            device: torch.device, warmup: int, repeats: int,
            prepare: Callable[[dict], dict] | None = None,
            autocast_dtype: torch.dtype | None = None) -> dict:
    training = mode == "training"
    encoder.train(training)

    def step() -> None:
        if training:
            encoder.zero_grad(set_to_none=True)
        run_batch = prepare(batch) if prepare else batch
        if training:
            with torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else nullcontext():
                output = encoder(run_batch)
                # A fixed, nonzero objective exercises every candidate and shared prefix.
                valid = output[run_batch["cand_mask"]].float()
                loss = valid.square().mean() + valid.mean()
            loss.backward()
        else:
            with torch.inference_mode(), (torch.autocast(device.type, dtype=autocast_dtype)
                                          if autocast_dtype else nullcontext()):
                encoder(run_batch)

    cleanup(encoder, device)
    for _ in range(warmup):
        step()
    sync(device)
    encoder.zero_grad(set_to_none=True)
    starting_allocated = None
    if device.type == "cuda":
        starting_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        step()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    result = {"status": "ok", **summarize_times(samples)}
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        result.update(
            cuda_start_allocated_bytes=starting_allocated,
            cuda_peak_allocated_bytes=peak,
            cuda_peak_increment_bytes=peak - starting_allocated,
        )
    else:
        result.update(cuda_start_allocated_bytes=None, cuda_peak_allocated_bytes=None,
                      cuda_peak_increment_bytes=None)
    encoder.zero_grad(set_to_none=True)
    return result


def probe(encoder: nn.Module, batch: dict, reference: torch.Tensor | None,
          atol: float, rtol: float,
          autocast_dtype: torch.dtype | None = None) -> tuple[dict, torch.Tensor]:
    """Count the actual token-wise work without hooks affecting timed samples."""
    counts: list[int] = []

    def count_tokens(module: nn.Module, args: tuple) -> None:
        counts.append(args[0].numel())

    handle = encoder.backbone.get_input_embeddings().register_forward_pre_hook(count_tokens)
    try:
        encoder.eval()
        device_type = batch["input_ids"].device.type
        with torch.inference_mode(), (torch.autocast(device_type, dtype=autocast_dtype)
                                      if autocast_dtype else nullcontext()):
            actual = encoder(batch).float().cpu()
    finally:
        handle.remove()
    result = {"actual_backbone_tokens": sum(counts), "backbone_embedding_calls": len(counts),
              "backbone_tokens_per_call": counts}
    if reference is not None:
        # Ignore padded candidate slots when computing aggregate errors.
        valid = batch["cand_mask"].cpu()
        actual_valid, reference_valid = actual[valid], reference[valid]
        delta = (actual_valid - reference_valid).abs()
        tolerance = atol + rtol * reference_valid.abs()
        result["output_parity"] = {
            "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "normalized_rmse": float(delta.square().mean().sqrt()
                                     / reference_valid.square().mean().sqrt().clamp_min(1e-12)),
            "outside_tolerance_fraction": float((delta > tolerance).float().mean()),
            "atol": atol, "rtol": rtol,
            "allclose": bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
            "all_finite": bool(torch.isfinite(actual).all()),
        }
    return result, actual


def error_record(error: Exception) -> dict:
    return {"status": "oom" if isinstance(error, torch.OutOfMemoryError)
            or "out of memory" in str(error).lower() else "error",
            "error_type": type(error).__name__, "message": str(error)}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Local Qwen3 weights; default uses random weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--autocast-dtype", choices=("float16", "bfloat16"),
                        help="Compute dtype; --dtype controls parameter storage")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512, help="Random model only")
    parser.add_argument("--layers", type=int, default=4, help="Random model only")
    parser.add_argument("--length-scale", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--modes", nargs="+", choices=("inference", "training"),
                        default=["inference", "training"])
    parser.add_argument("--output", type=Path, default=Path("outputs/tree_benchmark.json"))
    args = parser.parse_args()
    for name in ("repeats", "threads", "hidden_size", "layers", "length_scale"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.hidden_size % 8:
        parser.error("warmup must be nonnegative; hidden-size must be divisible by 8")
    return args


def main() -> None:
    args = parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    autocast_dtype = getattr(torch, args.autocast_dtype) if args.autocast_dtype else None
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.model_path:
        backbone = Qwen3Model.from_pretrained(
            str(args.model_path), local_files_only=True, torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        model_source = str(args.model_path.resolve())
    else:
        config = Qwen3Config(
            vocab_size=4096, hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 3, num_hidden_layers=args.layers,
            num_attention_heads=4, num_key_value_heads=2, head_dim=args.hidden_size // 4,
            max_position_embeddings=max(4096, 256 * args.length_scale),
            attention_dropout=0.0, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        backbone = Qwen3Model(config)
        model_source = "random_qwen3"
    if backbone.config.attention_dropout != 0.0:
        raise ValueError("Benchmark parity requires attention_dropout=0")
    backbone.to(device=device, dtype=dtype)
    # Share the same parameters; no optimizer updates take place.
    encoders = {"path": PathEncoder(backbone), "tree": TreeEncoder(backbone)}
    prepare = {
        "path": lambda batch: batch_to_device(batch, device),
        "tree": lambda batch: batch_to_device(pack_tree_batch(batch), device),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("encoders.py", "data.py", "benchmark_tree.py")},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers_version,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "dtype": args.dtype, "autocast_dtype": args.autocast_dtype,
            "threads": args.threads, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "model": {"source": model_source, "parameters": sum(p.numel() for p in backbone.parameters()),
                  "hidden_size": backbone.config.hidden_size, "layers": backbone.config.num_hidden_layers,
                  "attention_heads": backbone.config.num_attention_heads,
                  "key_value_heads": backbone.config.num_key_value_heads},
        "settings": {"warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
                     "length_scale": args.length_scale, "modes": args.modes},
        "measurement_scope": (
            "Path vs final Tree, synchronized wall-clock medians. Both include H2D from CPU token batches; "
            "Tree also includes CPU packing on every step. Training includes zero_grad, scalar loss and "
            "backward; excludes optimizer, tokenization, data loading and scoring head. Memory is CUDA "
            "allocated peak and increment, not reserved memory. Same weights, dropout disabled."
        ),
        "scenarios": [],
    }
    compute_dtype = autocast_dtype or dtype
    atol, rtol = ((3e-2, 3e-2) if compute_dtype == torch.bfloat16 else
                  (5e-3, 5e-3) if compute_dtype == torch.float16 else (2e-5, 2e-4))
    failed = False
    for scenario_name in args.scenarios:
        spec = SCENARIOS[scenario_name]
        # Keep a scenario's input independent of CLI selection order.
        seed = args.seed + list(SCENARIOS).index(scenario_name)
        cpu_batch = make_batch(spec, backbone.config.vocab_size, seed, args.length_scale)
        scenario = {"name": scenario_name, "spec": spec,
                    "valid_path_tokens": int(cpu_batch["attention_mask"].sum()),
                    "implementations": {}}
        report["scenarios"].append(scenario)
        reference = None
        print(f"\n{scenario_name}", flush=True)
        for name, encoder in encoders.items():
            result = {}
            scenario["implementations"][name] = result
            try:
                probe_batch = prepare[name](cpu_batch)
                result["selected_backend"] = ("path" if name == "path" else
                                               encoder._select_backend(probe_batch["tree_meta"]))
                details, actual = probe(encoder, probe_batch, reference, atol, rtol, autocast_dtype)
                result.update(details)
                if name == "path":
                    reference = actual
                else:
                    failed |= not result.get("output_parity", {}).get("allclose", False)
                del actual, probe_batch
            except (RuntimeError, ValueError) as error:
                result["probe"] = error_record(error)
                failed = True
                cleanup(backbone, device)
            for mode in args.modes:
                try:
                    result[mode] = measure(encoder, cpu_batch, mode, device,
                                           args.warmup, args.repeats, prepare[name], autocast_dtype)
                    baseline = scenario["implementations"].get("path", {}).get(mode, {})
                    if name == "tree" and baseline.get("status") == "ok":
                        result[mode]["speedup_vs_path"] = baseline["median_ms"] / result[mode]["median_ms"]
                except (RuntimeError, ValueError) as error:
                    result[mode] = error_record(error)
                    failed = True
                timing = result[mode]
                message = (f"{timing['median_ms']:.3f} ms" if timing["status"] == "ok"
                           else f"{timing['status']}: {timing.get('message', '')}")
                print(f"  {name:4s} {mode:9s} {message}", flush=True)
            cleanup(backbone, device)
            write_report(args.output, report)
    report["passed"] = not failed
    write_report(args.output, report)
    print(f"\nJSON report: {args.output.resolve()}")
    if failed:
        raise SystemExit("Benchmark recorded an execution error or output-parity failure; see JSON")


if __name__ == "__main__":
    main()
