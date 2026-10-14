"""Fit per-primitive temperatures on the Typed Decisions calibration split, as v1 did.

The logits come from the serving engine itself (same encoder, same BF16 weights), before any
temperature; the fit is typed_decisions.experiment.calibrate, the grid v1 used.

usage: python coding_gates/calibrate.py --checkpoint CKPT --model-path DIR --out temperatures.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLIT = ROOT / "typed_decisions" / "prepared"


def raw_logits(engine, requests, batch=20):
    """Per-question logits in request order, exactly as DecisionEngine.evaluate computes them."""
    import torch
    from jev_service.contract import encode_paths, prepare
    from jev_service.prefix import encode
    out = []
    for i in range(0, len(requests), batch):
        prepared = prepare({"requests": requests[i:i + batch]})
        paths, locations, rows = encode_paths(prepared, engine.tokenizer, engine.max_tokens)
        states = [si for si, request in enumerate(prepared) for _ in request["questions"]]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vectors, mask, _ = encode(engine, paths, locations, rows, engine.encoder, states)
            logits = engine.model._score(vectors, mask).float()
        out += [z[:len(q["candidates"])] for q, z in zip(rows, logits.cpu().tolist())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from jev_service.engine import DecisionEngine
    from typed_decisions.experiment import calibrate
    engine = DecisionEngine(args.checkpoint, args.model_path)
    requests = [json.loads(l) for l in open(SPLIT / "calibration_requests.jsonl", encoding="utf-8")]
    gold = {g["id"]: g for g in map(json.loads, open(SPLIT / "calibration_questions.jsonl", encoding="utf-8"))}
    order = [gold[f"{r['id']}:{q['id']}"] for r in requests for q in r["questions"]]
    rows = [{"type": g["question"]["type"], "target": g["target"], "logits": z}
            for g, z in zip(order, raw_logits(engine, requests))]
    assert len(rows) == len(order)
    temperatures = calibrate(rows)
    args.out.write_text(json.dumps(temperatures, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v["temperature"], 4) for k, v in temperatures.items()}))


if __name__ == "__main__":
    main()
