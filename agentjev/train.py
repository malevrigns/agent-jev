"""Single-GPU BF16 training with epoch budgets, dev selection and JSONL logs.

The optimizer budget is epochs * ceil(batches_per_epoch / grad_accum).
Use --max-steps for diagnostic overrides; the legacy config key is rejected.
Non-finite losses or gradients discard the accumulation window. Dev evaluation
runs only at optimizer-step boundaries and selects best.pt by soft cross-entropy.

Usage: python -m agentjev.train --config configs/tree_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from agentjev.data import (AgentJevDataset, EpochMixer, batch_to_device,
                           make_collate, pin_batch_memory)
from agentjev.losses import compute_losses
from agentjev.model import AgentJevModel
# Re-export helpers so existing imports from agentjev.train keep working.
from agentjev.train_eval import (
    _expected_gold, _save_ckpt, _save_json, _verify_alignment,
    evaluate_dev, metrics, probabilities, report, run_dev_eval,
)
from agentjev.train_monitor import WandbLogger, _flatten, init_wandb


def build_param_groups(model: AgentJevModel, lr_cfg: dict) -> list[dict]:
    """Layer-wise LR decay groups: embedding / bottom 12 / middle 8 /
    top 8 (incl. final norm) / decision head."""
    groups = {"embed": [], "bottom": [], "middle": [], "top": [], "head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".backbone." not in name:
            groups["head"].append(p)
        elif "embed_tokens" in name:
            groups["embed"].append(p)
        elif ".layers." in name:
            idx = int(name.split(".layers.")[1].split(".")[0])
            if idx < 12:
                groups["bottom"].append(p)
            elif idx < 20:
                groups["middle"].append(p)
            else:
                groups["top"].append(p)
        else:  # backbone.norm etc.
            groups["top"].append(p)
    return [{"params": groups[k], "lr": lr_cfg[k], "name": k}
            for k in ("embed", "bottom", "middle", "top", "head")
            if groups[k]]


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _dry_run(cfg, mixer, collate, *, steps_per_epoch, total_steps, progress):
    """Validate packing and source interleaving without loading the model."""
    print("[dry_run] config loaded; sampling 3 batches through the mixer "
          "and collate (no model, no training)")
    if mixer is None:
        raise SystemExit("[dry_run] config has no data_sources; nothing to mix")
    for b in range(3):
        src_name, states_b = mixer.next_batch()
        batch = collate(states_b)
        print(f"[dry_run] batch {b}: source={src_name} "
              f"states={batch['n_states']} questions={batch['n_questions']} "
              f"paths={batch['n_paths']} dropped={batch['n_dropped']} "
              f"trunc_states={batch['n_trunc_states']} "
              f"trunc_cands={batch['n_trunc_cands']}")
        tm = batch.get("tree_meta")
        if tm:
            print(f"[dry_run]   tree_meta: mode={tm['mode']} "
                  f"layout={tm.get('layout', '-')} "
                  f"input_tokens={tm['input_tokens']} "
                  f"unique={tm.get('unique_tokens', '-')} "
                  f"path_tokens={tm['path_tokens']} "
                  f"mask_mb={tm['mask_bytes'] / 2**20:.2f}"
                  + (f" fallback={tm['fallback_reason']}"
                     if tm.get("fallback_reason") else ""))
        if batch["dropped"]:
            print(f"[dry_run]   dropped detail: {batch['dropped'][:3]}")
    st = mixer.stats()
    print(f"[dry_run] epoch plan: batches/source={st['batches_per_source']} "
          f"batches/epoch={st['batches_per_epoch']} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps}")
    n_sim = min(300, st["batches_per_epoch"])
    counts = {name: 0 for name in mixer.names}
    for _ in range(n_sim):
        name, states_b = mixer.next_batch()
        counts[name] += len(states_b)
    tot = sum(counts.values())
    tot_w = sum(mixer.weights)
    print(f"[dry_run] prefix interleaving over the first {n_sim} batches "
          f"({tot} states):")
    for (name, w) in zip(mixer.names, mixer.weights):
        print(f"  {name}: target={w/tot_w:.3f} achieved={counts[name]/tot:.3f}")
    # A dry run must not touch the network, so wandb is described here
    # rather than connected to.
    wb_block = cfg.get("wandb") or {}
    if isinstance(wb_block, bool):
        wb_block = {"enabled": wb_block}
    bar_state = ("on (auto-off when stderr is not a TTY)" if progress
                 else "OFF (progress: false)")
    if wb_block.get("enabled", False):
        mode = wb_block.get("mode") or "auto (uploads with a key, else local)"
        wb_state = (f"project={wb_block.get('project', 'agentjev')} "
                    f"mode={mode} "
                    f"name={wb_block.get('run_name') or '<config>-<timestamp>'}")
    else:
        wb_state = "OFF (wandb.enabled is false)"
    print(f"[dry_run] progress={bar_state} | wandb={wb_state}")
    print("[dry_run] OK, exiting before model load")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="validate config + source mixer + collate on a few "
                         "batches, then exit before model load/training")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="diagnostic override that caps total_steps and stops "
                         "the run mid-epoch (e.g. a VRAM probe). Deliberately "
                         "not a config key, so a probe cannot be mistaken for "
                         "a training protocol; recorded in the checkpoints.")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Without this guard a config still carrying the old key would silently
    # run a full epoch instead of the few steps it used to mean.
    if "max_steps" in cfg:
        raise SystemExit(
            "[fatal] cfg['max_steps'] was replaced by cfg['epochs']; the "
            "step budget is now derived from the data. Use the --max-steps "
            "CLI flag for a diagnostic short run.")

    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.get("device", "cuda:0"))
    out_dir = cfg["out_dir"]
    log_path = cfg["log_path"]
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if cfg.get("log_reset", False):
        # A stale selection.json from a previous run in the same out_dir
        # would describe a checkpoint this run never produced. best.pt is
        # deliberately never removed: it is the one artifact worth keeping.
        for stale in (log_path, os.path.join(out_dir, "selection.json")):
            if os.path.exists(stale):
                os.remove(stale)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate = make_collate(tokenizer, max_len=cfg.get("max_len", 512),
                           max_state_tokens=cfg.get("max_state_tokens", 256),
                           encoder_impl=cfg.get("encoder_impl", "path"),
                           tree_max_mask_bytes=cfg.get("tree_max_mask_bytes", 64 * 1024 * 1024),
                           tree_attention_impl=cfg.get("tree_attention_impl", "auto"))
    pin_memory = device.type == "cuda" and cfg.get("pin_memory", True)
    mixer = None
    loader = None
    if cfg.get("data_sources"):
        mixer = EpochMixer(cfg["data_sources"], cfg["batch_states"],
                           seed=seed + 1)
        print("data sources:")
        for name, n, w, r in zip(mixer.names, mixer.sizes, mixer.weights,
                                 mixer.repeats):
            print(f"  {name}: {n} states, weight {w}, repeat {r}")
    else:
        dataset = AgentJevDataset(cfg["data_path"])
        loader = DataLoader(dataset, batch_size=cfg["batch_states"], shuffle=True,
                            collate_fn=collate, drop_last=False,
                            num_workers=0, pin_memory=pin_memory)

    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise SystemExit(f"[fatal] epochs must be >= 1, got {epochs}")
    batches_per_epoch = (mixer.batches_per_epoch if mixer is not None
                         else len(loader))
    if batches_per_epoch == 0:
        raise SystemExit("[fatal] no batches per epoch; is the data empty?")
    steps_per_epoch = -(-batches_per_epoch // grad_accum)
    total_steps = steps_per_epoch * epochs
    max_steps_override = args.max_steps
    if max_steps_override is not None:
        if max_steps_override < 1:
            raise SystemExit("--max-steps must be >= 1")
        print(f"[warn] --max-steps {max_steps_override} overrides the derived "
              f"total_steps={total_steps} (epochs={epochs}); the run will "
              f"stop mid-epoch")
        total_steps = max_steps_override
    print(f"epochs={epochs} batches/epoch={batches_per_epoch} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"grad_accum={grad_accum} "
          f"effective_batch={cfg['batch_states'] * grad_accum} states")

    # ---- Dev split. Loaded before the model so a bad path fails early.
    eval_path = cfg.get("eval_path", "data/agentjev/dev.jsonl")
    eval_every = int(cfg.get("eval_every", 50))
    eval_enabled = bool(eval_every > 0 and eval_path)
    eval_states: list[dict] = []
    eval_batch_states = int(cfg.get("eval_batch_states", cfg["batch_states"]))
    eval_verify = bool(cfg.get("eval_verify_align", True))
    save_best = bool(cfg.get("save_best", True))
    if eval_enabled:
        eval_ds = AgentJevDataset(eval_path)
        eval_states = [eval_ds[i] for i in range(len(eval_ds))]  # file order
        if not eval_states:
            raise SystemExit(f"[fatal] empty eval_path: {eval_path}")
        # Selection now happens inside training, so a split overlap would
        # invalidate the whole protocol. Cheap to assert, so assert it.
        train_ids = {s.get("id") for ds in
                     ([d for d in mixer.datasets] if mixer is not None
                      else [dataset])
                     for s in (ds[i] for i in range(len(ds)))}
        overlap = train_ids & {s.get("id") for s in eval_states}
        if overlap:
            raise SystemExit(f"[fatal] eval split overlaps train: "
                             f"{sorted(overlap)[:3]}")
        print(f"dev eval: {len(eval_states)} states from {eval_path} "
              f"every {eval_every} steps, batch_states={eval_batch_states}")
    else:
        print("dev eval: DISABLED (eval_every=0 or empty eval_path); "
              "no best.pt and no selection.json will be written")

    effective = {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batches_per_epoch": batches_per_epoch,
        "total_steps": total_steps,
        "grad_accum": grad_accum,
        "batch_states": cfg["batch_states"],
        "warmup_steps": max(1, int(total_steps * cfg.get("warmup_ratio", 0.02))),
        "eval_every": eval_every,
        "eval_batch_states": eval_batch_states,
        "eval_path": eval_path if eval_enabled else None,
        "max_steps_override": max_steps_override,
    }
    progress = bool(cfg.get("progress", True))

    if args.dry_run:
        _dry_run(cfg, mixer, collate, steps_per_epoch=steps_per_epoch,
                 total_steps=total_steps, progress=progress)
        return

    # wandb is brought up BEFORE the model so a missing package, a missing
    # login or an unreachable server costs seconds instead of a 0.6B weight
    # load plus a VRAM allocation.
    wb = WandbLogger(init_wandb(cfg, config_path=args.config, out_dir=out_dir,
                                effective=effective, total_steps=total_steps))

    dtype_name = cfg.get("dtype", "float32")
    if not hasattr(torch, dtype_name):
        raise SystemExit(f"[fatal] unknown dtype in config: {dtype_name!r}")
    model_dtype = getattr(torch, dtype_name)

    model = AgentJevModel(
        cfg["model_path"],
        set_dim=cfg.get("set_dim", 256),
        set_layers=cfg.get("set_layers", 2),
        set_heads=cfg.get("set_heads", 4),
        encoder_impl=cfg.get("encoder_impl", "path"),
        dtype=model_dtype,
        tree_max_mask_bytes=cfg.get("tree_max_mask_bytes", 64 * 1024 * 1024),
        tree_attention_impl=cfg.get("tree_attention_impl", "auto"),
    ).to(device)
    model.train()

    if cfg.get("freeze_backbone", False):
        model.path_encoder.backbone.requires_grad_(False)
        print("backbone frozen: training the candidate head only")

    init_from = cfg.get("init_from")
    if init_from:
        # weights only: optimizer/scheduler state is intentionally not
        # restored (fresh phase-2 optimization from phase-1 weights).
        ck = torch.load(init_from, map_location="cpu")
        model.load_state_dict(ck.get("state_dict", ck), strict=True)
        print(f"initialized weights from {init_from} (step {ck.get('step')})")

    param_groups = build_param_groups(model, cfg["lr"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.01))
    # total_steps was derived from the data above; --max-steps may already
    # have capped it, so the cosine schedule always matches the real run.
    scheduler = build_scheduler(optimizer, total_steps,
                                cfg.get("warmup_ratio", 0.02))
    grad_clip = cfg.get("grad_clip", 1.0)
    log_every = cfg.get("log_every", 1)
    max_consec_fail = cfg.get("max_consec_fail", 25)
    w_cfg = cfg.get("loss_weights", {})
    perm_reg = w_cfg.get("perm_kl", 0.0) > 0

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # LambdaLR runs one step() at construction, which rewrites group['lr'] to
    # base_lr * lr_lambda(0) -- zero during warmup. Read the configured lr off
    # scheduler.base_lrs instead, or this prints 0.0 for every group.
    print(f"params: {n_params/1e6:.1f}M ({n_trainable/1e6:.1f}M trainable) "
          f"| dtype: {dtype_name} | groups: "
          + ", ".join(f"{g['name']}={lr}"
                      for g, lr in zip(param_groups, scheduler.base_lrs)))

    step = 0
    win_micro = 0          # valid microbatches in the CURRENT accum window
    skipped = 0
    consec_fail = 0
    t0 = time.time()
    win_states = 0
    win_tokens = 0
    win_questions = 0
    win_sources: dict[str, int] = {}
    win_terms: dict[str, float] = {}
    win_n_micro = 0        # valid microbatches since last log event
    win_dropped = 0
    win_trunc_states = 0
    win_trunc_cands = 0
    win_tree_modes: dict[str, int] = {}
    win_tree_fallbacks: dict[str, int] = {}
    if device.type == "cuda":
        # Measures the run, not the process: opt-in peak tracking so a config
        # can be sized against real numbers instead of an estimate.
        torch.cuda.reset_peak_memory_stats()
    total_states = 0
    total_questions = 0
    total_tokens = 0
    total_dropped = 0
    win_source_mix: dict[str, int] = {}
    loader_iter = iter(loader) if loader is not None else None
    # Selection state. best["step"] is None until the first dev pass, which
    # is why the baseline is always written (see run_dev_eval).
    best: dict = {"ce": float("inf"), "step": None, "epoch": 0.0}
    history: list[dict] = []
    best_ce_init: float | None = None
    last_eval_questions = 0
    last_eval_dropped = 0
    # Peak memory is tracked on the Python side so that evaluating does not
    # inflate the training-only number reported as peak_mem_gb.
    train_peak_gb = 0.0
    dev_peak_max = 0.0

    bar = None

    def emit(msg: str) -> None:
        """Print without garbling an active progress bar."""
        if bar is not None:
            bar.write(msg)
        else:
            print(msg)

    def _dev_pass(step_now: int) -> dict:
        """Run one dev eval with the peak-memory bookkeeping around it."""
        nonlocal train_peak_gb, dev_peak_max
        nonlocal last_eval_questions, last_eval_dropped
        if device.type == "cuda":
            train_peak_gb = max(train_peak_gb,
                                torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()
        info = run_dev_eval(
            model, states=eval_states, collate=collate, device=device, cfg=cfg,
            batch_states=eval_batch_states, verify=eval_verify, out_dir=out_dir,
            step=step_now, steps_per_epoch=steps_per_epoch,
            total_steps=total_steps, effective=effective, best=best,
            history=history, save_best=save_best, save_reports=True)
        if device.type == "cuda":
            dev_peak_max = max(dev_peak_max, info["peak"] or 0.0)
            torch.cuda.reset_peak_memory_stats()
        last_eval_questions = info["stats"]["questions"]
        last_eval_dropped = info["stats"]["dropped"]
        m = info["metrics"]
        flag = " [best]" if info["improved"] else ""
        inc = (f", peak={info['peak']:.3f}GB +{info['increment']:.3f}"
               if info["peak"] is not None else "")
        emit(f"  dev@{step_now}: ce={m['soft_cross_entropy']:.4f} "
             f"acc={m['accuracy']:.4f} brier={m['brier_sum']:.4f} "
             f"({info['seconds']}s, {info['stats']['questions']} q{flag}{inc})")
        return info

    optimizer.zero_grad(set_to_none=True)

    if eval_enabled and cfg.get("eval_at_init", True):
        init_info = _dev_pass(0)
        best_ce_init = init_info["metrics"]["soft_cross_entropy"]
        # The baseline is a real data point, not just a threshold to beat:
        # without it on the curve there is no way to see what training
        # actually bought.
        wb.log({"step": 0, "dev": init_info["metrics"],
                "is_best": init_info["improved"],
                "eval_seconds": init_info["seconds"]}, step=0)

    # Start throughput timing after the initialization dev pass.
    win_t = time.time()

    if progress:
        bar = tqdm(total=total_steps, initial=step, desc="train", unit="step",
                   dynamic_ncols=True, disable=None,
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                              "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                              "{postfix}")

    while step < total_steps:
        src_name = None
        if mixer is not None:
            src_name, states_b = mixer.next_batch()
            batch = collate(states_b)
        else:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
        if mixer is not None and pin_memory:
            batch = pin_batch_memory(batch)
        batch = batch_to_device(batch, device, non_blocking=pin_memory)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, perm_reg=perm_reg)
            terms = compute_losses(outputs, batch, w_cfg,
                                   margin=cfg.get("margin", 0.5))
            loss = terms["total"] / grad_accum

        if not torch.isfinite(terms["total"]):
            # Discard the whole accumulation window and restart it, so a
            # step is only ever taken on grad_accum valid microbatches.
            skipped += 1
            consec_fail += 1
            win_micro = 0
            optimizer.zero_grad(set_to_none=True)
            emit(f"[warn] non-finite loss, restarting accum window "
                 f"(skipped total: {skipped}, consecutive: {consec_fail})")
            if consec_fail >= max_consec_fail:
                ckpt = os.path.join(out_dir, "circuit_break.pt")
                torch.save({"state_dict": model.state_dict(), "config": cfg,
                            "step": step}, ckpt)
                raise SystemExit(
                    f"[fatal] {consec_fail} consecutive non-finite batches; "
                    f"saved {ckpt} and aborting")
            continue

        loss.backward()

        # Only count data that actually contributed to the window.
        win_micro += 1
        win_n_micro += 1
        n_tok = batch["n_valid_tokens"]
        win_states += batch["n_states"]
        win_tokens += n_tok
        win_questions += batch["n_questions"]
        win_dropped += batch.get("n_dropped", 0)
        win_trunc_states += batch.get("n_trunc_states", 0)
        win_trunc_cands += batch.get("n_trunc_cands", 0)
        tm = batch.get("tree_meta")
        if tm:
            # Only present when encoder_impl="tree"; records which backend the
            # TreeEncoder actually routed this batch to ("tree" vs a "path"
            # fallback) so a run can prove the tree branch was exercised.
            win_tree_modes[tm["mode"]] = win_tree_modes.get(tm["mode"], 0) + 1
            reason = tm.get("fallback_reason")
            if reason:
                win_tree_fallbacks[reason] = win_tree_fallbacks.get(reason, 0) + 1
        for env, c in batch["sources"].items():
            win_sources[env] = win_sources.get(env, 0) + c
        if src_name is not None:
            win_source_mix[src_name] = (win_source_mix.get(src_name, 0)
                                        + batch["n_states"])
        total_states += batch["n_states"]
        total_questions += batch["n_questions"]
        total_tokens += n_tok
        total_dropped += batch.get("n_dropped", 0)
        for k, v in terms.items():
            win_terms[k] = win_terms.get(k, 0.0) + float(v.detach())
        consec_fail = 0

        if win_micro % grad_accum != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            skipped += 1
            consec_fail += 1
            win_micro = 0
            optimizer.zero_grad(set_to_none=True)
            emit(f"[warn] non-finite grad norm at step {step + 1}, window "
                 f"discarded (skipped total: {skipped}, consecutive: {consec_fail})")
            if consec_fail >= max_consec_fail:
                ckpt = os.path.join(out_dir, "circuit_break.pt")
                torch.save({"state_dict": model.state_dict(), "config": cfg,
                            "step": step}, ckpt)
                raise SystemExit(
                    f"[fatal] {consec_fail} consecutive non-finite grad norms; "
                    f"saved {ckpt} and aborting")
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        win_micro = 0
        # The bar advances exactly once per OPTIMIZER STEP, at the only
        # place `step` changes, so it can never disagree with step/total.
        if bar is not None:
            bar.update(1)

        # Dev eval at an optimizer-step boundary. The `continue`s above (an
        # unfilled accumulation window, or a discarded NaN window) all skip
        # this point, so an eval can never land mid-window.
        dev_info = None
        if eval_enabled and (step % eval_every == 0 or step == total_steps):
            dev_info = _dev_pass(step)

        if (step % log_every == 0 or step == total_steps
                or dev_info is not None):
            now = time.time()
            dt = now - win_t
            lrs = scheduler.get_last_lr()
            denom = max(1, win_n_micro)
            mean_terms = {k: v / denom for k, v in win_terms.items()}
            event = {
                "step": step,
                "loss": mean_terms.get("total"),
                **{k: v for k, v in mean_terms.items() if k != "total"},
                "lr": max(lrs),
                "lr_head": lrs[-1],
                "grad_norm": float(grad_norm),
                "skipped": skipped,
                "dropped_questions": total_dropped,
                "trunc_states": win_trunc_states,
                "trunc_cands": win_trunc_cands,
                "tree_modes": dict(win_tree_modes),
                "tree_fallbacks": dict(win_tree_fallbacks),
                "sources": dict(win_sources),
                "source_mix": dict(win_source_mix),
                "samples_trained": total_questions,
                "states_trained": total_states,
                "tokens_total": total_tokens,
                "states_per_s": round(win_states / max(dt, 1e-9), 2),
                "questions_per_s": round(win_questions / max(dt, 1e-9), 2),
                "tokens_per_s": round(win_tokens / max(dt, 1e-9), 1),
                "elapsed_s": round(now - t0, 2),
                # Training-only high-water mark: reset_peak_memory_stats()
                # runs around every dev eval, so an eval's own allocation
                # cannot inflate this.
                "peak_mem_gb": round(max(train_peak_gb,
                                         torch.cuda.max_memory_allocated() / 1e9), 3)
                if device.type == "cuda" else None,
                # Epoch accounting, so a dashboard can render progress.
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "epoch": round(step / steps_per_epoch, 4),
                "best_ce": None if best["step"] is None else best["ce"],
                "best_step": best["step"],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if dev_info is not None:
                # "dev" is exactly metrics()-shaped, matching the key
                # scripts/train.py writes, so runs stay comparable. The
                # eval's own cost/accounting stays at the top level.
                event["dev"] = dev_info["metrics"]
                event["is_best"] = dev_info["improved"]
                event["eval_seconds"] = dev_info["seconds"]
                event["eval_states"] = dev_info["stats"]["states"]
                event["eval_questions"] = dev_info["stats"]["questions"]
                event["eval_dropped"] = dev_info["stats"]["dropped"]
                event["eval_tree_modes"] = dev_info["stats"]["tree_modes"]
                event["eval_peak_mem_gb"] = (round(dev_info["peak"], 3)
                                             if dev_info["peak"] is not None
                                             else None)
                event["eval_peak_increment_gb"] = (
                    round(dev_info["increment"], 3)
                    if dev_info["increment"] is not None else None)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            wb.log(_flatten({k: v for k, v in event.items()
                             if k not in ("step", "ts")}), step=step)
            if bar is not None and not bar.disable:
                bar.set_postfix(loss=f"{event['loss']:.4f}",
                                lr=f"{event['lr']:.1e}",
                                tok_s=event["tokens_per_s"])
            else:
                emit(f"step {step}/{total_steps} loss={event['loss']:.4f} "
                     f"lr={event['lr']:.2e} tok/s={event['tokens_per_s']}")
            win_states = 0
            win_tokens = 0
            win_questions = 0
            win_sources = {}
            win_source_mix = {}
            win_terms = {}
            win_n_micro = 0
            win_dropped = 0
            win_trunc_states = 0
            win_trunc_cands = 0
            win_tree_modes = {}
            win_tree_fallbacks = {}
            win_t = now

        save_every = cfg.get("save_every_steps", 0)
        if save_every > 0 and step % save_every == 0:
            ckpt = os.path.join(out_dir, f"checkpoint-{step}.pt")
            _save_ckpt(ckpt, {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
                "step": step,
                "epoch": step / steps_per_epoch,
                "effective": effective,
            })
            emit(f"saved {ckpt}")

    # Close the bar before the tail output, so the final lines are not
    # interleaved with a repaint. leave=True keeps the finished bar on
    # screen; dropping the reference makes every later emit() a plain print.
    if bar is not None:
        bar.close()
        bar = None

    if cfg.get("save_final", True):
        ckpt = os.path.join(out_dir, "final.pt")
        _save_ckpt(ckpt, {
            "state_dict": model.state_dict(),
            "config": cfg,
            "step": step,
            "epoch": step / steps_per_epoch,
            "effective": effective,
        })
        emit(f"saved {ckpt}")

    elapsed = time.time() - t0
    selection = None
    if eval_enabled:
        # A run too short to reach eval_every (with eval_at_init off) leaves
        # best untouched; json.dump would then write a bare `Infinity`, which
        # is not valid JSON. Report null instead.
        selected = best["step"] is not None
        # Written last, so its existence means the run reached total_steps.
        selection = {
            "best_step": best["step"],
            "best_epoch": round(best["epoch"], 4) if selected else None,
            # Same key name scripts/train.py uses, for cross-run comparison.
            "dev_soft_cross_entropy": best["ce"] if selected else None,
            "dev_soft_cross_entropy_init": best_ce_init,
            # Never None when an eval ran: the baseline is always written
            # (see run_dev_eval), so best_step is 0 at worst.
            "checkpoint": (os.path.join(out_dir, "best.pt")
                           if selected and save_best else None),
            "selection_metric": "uncalibrated dev soft cross-entropy (temp 1.0)",
            "protocol": effective,
            "history": history,
            "training_seconds": round(elapsed, 2),
            "peak_mem_gb": round(train_peak_gb, 3) if device.type == "cuda" else None,
            "dev_peak_mem_gb_max": (round(dev_peak_max, 3)
                                    if device.type == "cuda" else None),
            "eval_states": len(eval_states),
            "eval_questions": last_eval_questions,
            "eval_dropped": last_eval_dropped,
            "selection_completed_before_test": True,
        }
        _save_json(os.path.join(out_dir, "selection.json"), selection)
        emit(f"saved {os.path.join(out_dir, 'selection.json')}")

    emit(f"done. steps={step} skipped_batches={skipped} "
         f"dropped_questions={total_dropped} elapsed={elapsed:.1f}s")

    # The selection is what a run is actually judged by, so it belongs in the
    # wandb SUMMARY (which survives in the run's table row) rather than only
    # scrolling past in the console log.
    summary = {
        "steps": step,
        "skipped_batches": skipped,
        "dropped_questions": total_dropped,
        "training_seconds": round(elapsed, 2),
        "peak_mem_gb": round(train_peak_gb, 3) if device.type == "cuda" else None,
        "dev_peak_mem_gb_max": (round(dev_peak_max, 3)
                                if device.type == "cuda" else None),
    }
    if selection is not None:
        summary.update({k: selection[k] for k in
                        ("best_step", "best_epoch", "dev_soft_cross_entropy",
                         "dev_soft_cross_entropy_init", "checkpoint")})
    wb.finish(summary)


if __name__ == "__main__":
    main()
