"""AgentJev training loop.

BF16 (autocast) full-parameter training on a single GPU. AdamW with
layer-wise LR decay groups, cosine schedule with warmup, grad clipping,
gradient accumulation, and JSONL event logging for the dashboard.

NaN guard (v2): the accumulation window keeps its own counter of VALID
microbatches. A non-finite loss or grad norm discards the whole window
and restarts it from zero, so an optimizer step is always taken on
exactly ``grad_accum`` valid microbatches. Consecutive failures are
circuit-broken at ``max_consec_fail``. Logged losses are window means,
not the last microbatch.

Usage:
  python -m agentjev.train --config configs/smoke.yaml
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
from transformers import AutoTokenizer

from agentjev.data import (AgentJevDataset, SourceMixer, batch_to_device,
                           make_collate, pin_batch_memory)
from agentjev.losses import compute_losses
from agentjev.model import AgentJevModel


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="validate config + source mixer + collate on a few "
                         "batches, then exit before model load/training")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    if cfg.get("log_reset", False) and os.path.exists(log_path):
        os.remove(log_path)

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
        mixer = SourceMixer(cfg["data_sources"], cfg["batch_states"],
                            seed=seed + 1)
        print("data sources:")
        for name, ds, w in zip(mixer.names, mixer.datasets, mixer.weights):
            print(f"  {name}: {len(ds)} states, weight {w}")
    else:
        dataset = AgentJevDataset(cfg["data_path"])
        loader = DataLoader(dataset, batch_size=cfg["batch_states"], shuffle=True,
                            collate_fn=collate, drop_last=False,
                            num_workers=0, pin_memory=pin_memory)

    if args.dry_run:
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
            if batch["dropped"]:
                print(f"[dry_run]   dropped detail: {batch['dropped'][:3]}")
        # proportion check over more draws (no collate, cheap)
        n_sim = 300
        counts = {name: 0 for name in mixer.names}
        for _ in range(n_sim):
            name, states_b = mixer.next_batch()
            counts[name] += len(states_b)
        tot = sum(counts.values())
        print(f"[dry_run] state-draw proportions over {n_sim} batches "
              f"({tot} states):")
        for (name, w) in zip(mixer.names, mixer.weights):
            print(f"  {name}: target={w:.3f} achieved={counts[name]/tot:.3f}")
        print("[dry_run] OK, exiting before model load")
        return

    model = AgentJevModel(
        cfg["model_path"],
        set_dim=cfg.get("set_dim", 256),
        set_layers=cfg.get("set_layers", 2),
        set_heads=cfg.get("set_heads", 4),
        encoder_impl=cfg.get("encoder_impl", "path"),
        dtype=torch.float32,
        tree_max_mask_bytes=cfg.get("tree_max_mask_bytes", 64 * 1024 * 1024),
        tree_attention_impl=cfg.get("tree_attention_impl", "auto"),
    ).to(device)
    model.train()

    init_from = cfg.get("init_from")
    if init_from:
        # weights only: optimizer/scheduler state is intentionally not
        # restored (fresh phase-2 optimization from phase-1 weights).
        ck = torch.load(init_from, map_location="cpu")
        model.load_state_dict(ck.get("state_dict", ck), strict=True)
        print(f"initialized weights from {init_from} (step {ck.get('step')})")

    param_groups = build_param_groups(model, cfg["lr"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.01))
    max_steps = cfg["max_steps"]
    grad_accum = cfg.get("grad_accum", 1)
    scheduler = build_scheduler(optimizer, max_steps, cfg.get("warmup_ratio", 0.02))
    grad_clip = cfg.get("grad_clip", 1.0)
    log_every = cfg.get("log_every", 1)
    max_consec_fail = cfg.get("max_consec_fail", 25)
    w_cfg = cfg.get("loss_weights", {})
    perm_reg = w_cfg.get("perm_kl", 0.0) > 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M | groups: "
          + ", ".join(f"{g['name']}={g['lr']}" for g in param_groups))

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
    win_t = t0
    total_states = 0
    total_questions = 0
    total_tokens = 0
    total_dropped = 0
    win_source_mix: dict[str, int] = {}
    loader_iter = iter(loader) if loader is not None else None

    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
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
            print(f"[warn] non-finite loss, restarting accum window "
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
            print(f"[warn] non-finite grad norm at step {step + 1}, window "
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

        if step % log_every == 0 or step == max_steps:
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
                "sources": dict(win_sources),
                "source_mix": dict(win_source_mix),
                "samples_trained": total_questions,
                "states_trained": total_states,
                "tokens_total": total_tokens,
                "states_per_s": round(win_states / max(dt, 1e-9), 2),
                "questions_per_s": round(win_questions / max(dt, 1e-9), 2),
                "tokens_per_s": round(win_tokens / max(dt, 1e-9), 1),
                "elapsed_s": round(now - t0, 2),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(f"step {step}/{max_steps} loss={event['loss']:.4f} "
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
            win_t = now

        save_every = cfg.get("save_every_steps", 0)
        if save_every > 0 and step % save_every == 0:
            ckpt = os.path.join(out_dir, f"checkpoint-{step}.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
                "step": step,
            }, ckpt)
            print(f"saved {ckpt}")

    if cfg.get("save_final", True):
        ckpt = os.path.join(out_dir, "final.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg,
            "step": step,
        }, ckpt)
        print(f"saved {ckpt}")

    print(f"done. steps={step} skipped_batches={skipped} "
          f"dropped_questions={total_dropped} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
