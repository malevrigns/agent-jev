"""Dev metrics, alignment checks and atomic checkpoint selection.

Metric definitions follow scripts/eval_agentjev.py; keep their numeric behavior
in sync. Row alignment comes from collate's q_meta, including dropped questions.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from agentjev.data import batch_to_device


def probabilities(logits, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temp
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def metrics(rows: list[dict]) -> dict:
    """照搬 scripts/eval_agentjev.py:42 的定义，probs 需已按温度缩放过。"""
    correct, ces, briers, soft, confidence, score_errors = [], [], [], [], [], []
    for r in rows:
        p = np.asarray(r["probs"])
        y = np.asarray(r["target"])
        guess, gold = int(p.argmax()), int(y.argmax())
        correct.append(float(guess == gold))
        ces.append(float(-(y * np.log(np.clip(p, 1e-12, 1))).sum()))
        briers.append(float(((p - y) ** 2).sum()))
        soft.append(float(y[guess]))
        confidence.append(float(p.max()))
        if r["ordinal"]:
            score_errors.append(abs(float(np.arange(len(p)) @ (p - y))))
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        ix = [
            i
            for i, c in enumerate(confidence)
            if lo <= c < (lo + 0.1 if lo < 0.9 else 1.0000001)
        ]
        if ix:
            ece += len(ix) / len(rows) * abs(
                np.mean([correct[i] for i in ix]) - np.mean([confidence[i] for i in ix])
            )
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "soft_cross_entropy": float(np.mean(ces)),
        "brier_sum": float(np.mean(briers)),
        "gold_mass_at_selected_option": float(np.mean(soft)),
        "ece_10_bins_vs_gold_argmax": float(ece),
        "score_expectation_mae": float(np.mean(score_errors)) if score_errors else None,
    }


def report(rows: list[dict]) -> dict:
    # Empty groups are skipped, not measured: np.mean([]) is nan and would
    # silently poison a by_type/by_workflow entry.
    def group(key):
        return {k: metrics(g) for k, g in
                ((k, [r for r in rows if r[key] == k])
                 for k in sorted({r[key] for r in rows})) if g}

    return {
        "overall": metrics(rows),
        "by_type": group("qtype"),
        "by_workflow": group("workflow"),
    }


def _expected_gold(gold: dict) -> list[float]:
    """Normalize a gold the same way data.make_collate does (data.py:566-574)."""
    if "counts" in gold:
        cv = [float(x) for x in gold["counts"]]
        t = sum(cv)
        return [x / t for x in cv] if t > 0 else cv
    dv = [float(x) for x in gold["distribution"]]
    s = sum(dv)
    return [x / max(s, 1e-8) for x in dv]


def _verify_alignment(batch: dict, chunk: list[dict]) -> None:
    """Verify row order, candidate counts and targets against collate metadata."""
    q_meta = batch["q_meta"]
    if len(q_meta) != batch["n_questions"]:
        raise SystemExit(
            f"[fatal] collate returned {len(q_meta)} q_meta entries for "
            f"{batch['n_questions']} rows; row mapping is unsafe")
    prev = (-1, -1)
    for i, meta in enumerate(q_meta):
        if meta is None:  # empty-batch placeholder row
            continue
        key = (meta["state_index"], meta["question_index"])
        if key <= prev:
            raise SystemExit(
                f"[fatal] collate row {i} is out of document order: "
                f"{key} after {prev}")
        prev = key
        q = chunk[meta["state_index"]]["questions"][meta["question_index"]]
        if meta["gold"] != q["gold"] or meta["id"] != chunk[meta["state_index"]].get("id", "?"):
            raise SystemExit(
                f"[fatal] collate row {i} q_meta does not match state "
                f"{meta['state_index']} question {meta['question_index']}")
        exp = _expected_gold(q["gold"])
        n = int(batch["cand_mask"][i].sum())
        if n != len(exp):
            raise SystemExit(
                f"[fatal] row {i}: {n} candidate slots but the gold has "
                f"{len(exp)} entries")
        got = batch["target_dist"][i, :n]
        if not torch.allclose(got, torch.tensor(exp, dtype=got.dtype), atol=1e-5):
            raise SystemExit(
                f"[fatal] row {i} target_dist != gold of state {key[0]} "
                f"question {key[1]}: {got.tolist()} vs {exp}")


def evaluate_dev(model, states: list[dict], collate, device, *,
                 batch_states: int, verify: bool = True):
    """Score states in file order with the training collate and BF16 autocast."""
    rows: list[dict] = []
    tree_modes: dict[str, int] = {}
    n_dropped = 0
    for start in range(0, len(states), batch_states):
        chunk = states[start:start + batch_states]
        # Collate OUTSIDE inference_mode: for encoder_impl="tree" it runs
        # CPU-side tree packing and mints real tensors, which must not
        # become inference tensors.
        batch = collate(chunk)
        tm = batch.get("tree_meta")
        if tm:
            tree_modes[tm["mode"]] = tree_modes.get(tm["mode"], 0) + 1
        if verify:
            _verify_alignment(batch, chunk)
        n_dropped += batch.get("n_dropped", 0)
        batch = batch_to_device(batch, device)
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch)["logits"].float().cpu().numpy()
        cand_mask = batch["cand_mask"].cpu().numpy()
        target = batch["target_dist"].cpu().numpy()
        for i, meta in enumerate(batch["q_meta"]):
            if meta is None:
                continue
            n = int(cand_mask[i].sum())
            rows.append({
                "id": meta["id"],
                "workflow": meta["workflow"],
                "qtype": meta["qtype"],
                "ordinal": meta["ordinal"],
                "target": target[i, :n].tolist(),
                "logits": logits[i, :n].tolist(),
                "probs": probabilities(logits[i, :n]).tolist(),
            })
    return rows, {"states": len(states), "questions": len(rows),
                  "dropped": n_dropped, "tree_modes": tree_modes}


def _save_ckpt(path: str, payload: dict) -> None:
    """Atomic checkpoint write: a reader never sees a partial file."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _save_json(path: str, value) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def run_dev_eval(model, *, states, collate, device, cfg, batch_states,
                 verify, out_dir, step, steps_per_epoch, total_steps,
                 effective, best, history, save_best, save_reports) -> dict:
    """Evaluate and select the best checkpoint, restoring model mode afterward.

    Training counters are untouched; the caller handles CUDA peak resets."""
    was_training = model.training
    # Resident bytes (weights + optimizer state) are not the eval's cost;
    # the marginal cost is what the eval stack adds on top of them. That is
    # the number that decides whether eval_batch_states can be raised.
    mem_entry = (torch.cuda.memory_allocated()
                 if device.type == "cuda" else 0)
    t_eval = time.time()
    try:
        rows, stats = evaluate_dev(model, states, collate, device,
                                   batch_states=batch_states, verify=verify)
    finally:
        model.train(was_training)
    dev_peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda" else None)
    dev_increment = (dev_peak - mem_entry / 1e9
                     if device.type == "cuda" else None)
    m = metrics(rows)
    epoch = step / steps_per_epoch
    # The baseline (step 0) is written unconditionally: with a strict "<"
    # comparison it can never beat itself, which is exactly how
    # scripts/train.py ends up with best_step == 0 and checkpoint None.
    improved = best["step"] is None or m["soft_cross_entropy"] < best["ce"]
    if improved:
        best.update(ce=m["soft_cross_entropy"], step=step, epoch=epoch)
        if save_best:
            _save_ckpt(os.path.join(out_dir, "best.pt"), {
                "state_dict": model.state_dict(),
                "config": cfg,
                "step": step,
                "epoch": epoch,
                "dev": m,
                "effective": effective,
                "role": "typed_decisions",
                "input_schema": "agentjev.decision.v1",
                "max_path_tokens": cfg.get("max_len", 512),
            })
    history.append({"step": step, "epoch": round(epoch, 4),
                    "soft_cross_entropy": m["soft_cross_entropy"],
                    "accuracy": m["accuracy"], "brier_sum": m["brier_sum"]})
    if save_reports:
        _save_json(os.path.join(out_dir,
                                "initial_dev.json" if step == 0
                                else f"dev_step_{step}.json"), report(rows))
    return {"metrics": m, "stats": stats, "peak": dev_peak,
            "increment": dev_increment,
            "seconds": round(time.time() - t_eval, 2), "improved": improved}
