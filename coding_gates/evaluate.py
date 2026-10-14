"""Score a checkpoint on the coding-gate splits, the README probes, and optionally Typed Decisions.

usage: python coding_gates/evaluate.py --checkpoint CKPT --model-path DIR [--temperatures T.json]
                                       [--splits dev test ood] [--paraphrase] [--typed dev] --out OUT.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

PROBES = [  # the README's own example, the guide's cases, and plain one-liners
    ("23 tests passed, 1 failed.", False), ("Tests run: 14, Failures: 1 — test_expired_token", False),
    ("Tests run: 14, Failures: 0, Errors: 0. BUILD SUCCESS", True),
    ("Compilation failed: UserAuthService.java:42: cannot find symbol verifyToken", False),
    ("No tests were executed (0 tests collected).", False), ("All 14 tests passed.", True),
    ("FAILED tests/test_auth.py::test_expired_token - AssertionError: assert 401 == 200", False),
    ("================ 14 passed in 0.31s ================", True),
]
# The gate asked in words the training data never uses (fixed before any fine-tuned model was scored).
PARAPHRASES = [
    {"id": "green", "type": "boolean", "question": "Is the test suite green?",
     "criteria": {"true": "Every test that ran passed, and at least one test ran.",
                  "false": "Something failed or errored, the suite could not load, or nothing ran."}},
    {"id": "zh", "type": "boolean", "question": "测试是否全部通过？",
     "criteria": {"true": "至少运行了一个测试，并且全部通过。", "false": "有测试失败或报错、测试无法加载，或者没有运行任何测试。"}},
    {"id": "done", "type": "boolean", "question": "Based on this test run, can the coding agent mark the task as done?",
     "criteria": {"true": "Yes: tests ran and all of them passed.",
                  "false": "No: a test failed or errored, the tests could not load, or no tests ran."}},
    {"id": "bare", "type": "boolean", "question": "Did all the tests pass?"},
]
# Fixed after the first fine-tune's dev results, before the second training run. Scored only on
# test and ood, once (--final-paraphrase). "inverted" asks whether something is wrong.
FINAL_PARAPHRASES = [
    {"id": "shows_pass", "type": "boolean", "question": "Do these results show a fully passing test run?",
     "criteria": {"true": "Yes, tests ran and none failed or errored.",
                  "false": "No, there are failures, errors, a load problem, or no tests."}},
    {"id": "zh_task_done", "type": "boolean", "question": "根据这次测试输出，任务可以算完成了吗？",
     "criteria": {"true": "可以：测试运行了，并且全部通过。", "false": "不可以：有失败或报错、无法加载，或者一个测试都没运行。"}},
    {"id": "red", "type": "boolean", "question": "Is this test run red?", "inverted": True,
     "criteria": {"true": "Red: something failed or errored, the tests could not load, or none ran.",
                  "false": "Not red: tests ran and every one passed."}},
    {"id": "zh_bare", "type": "boolean", "question": "所有测试都通过了吗？"},
    {"id": "terse", "type": "boolean", "question": "Tests passing?", "criteria": {"true": "yes", "false": "no"}},
]
PROBE_Q = {"id": "done", "type": "boolean", "question": "Are all tests passing?",
           "criteria": {"true": "The suite is green.", "false": "At least one test failed."}}


def metrics(rows):
    from typed_decisions.experiment import metrics as m
    return m(rows) if rows else None


def gate_errors(rows):
    gate = [r for r in rows if r["type"] == "boolean"]
    false_pass = [r for r in gate if r["target"][0] == 0 and r["probs"][0] >= .5]
    false_block = [r for r in gate if r["target"][0] == 1 and r["probs"][0] < .5]
    negatives = sum(r["target"][0] == 0 for r in gate)
    return {"n": len(gate), "negatives": negatives, "false_pass": len(false_pass),
            "false_pass_rate": len(false_pass) / max(1, negatives), "false_block": len(false_block)}


def score(engine, requests, gold, batch=20):
    rows = []
    for i in range(0, len(requests), batch):
        chunk = requests[i:i + batch]
        res = engine.evaluate({"requests": chunk})
        for req, out in zip(chunk, res["results"]):
            for q, a in zip(req["questions"], out["answers"]):
                g = gold[f"{req['id']}:{q['id']}"]
                rows.append({**g, "probs": list(a["distribution"].values())})
    return rows


def paraphrased(engine, requests, gold, keep, split, wordings=PARAPHRASES):
    """The gate in other words. Rows keep the "all passed" orientation, so an inverted
    wording's probabilities are flipped back before scoring."""
    out = {}
    for spec in wordings:
        q = {k: v for k, v in spec.items() if k != "inverted"}
        reqs = [{"id": r["id"], "state": r["state"], "questions": [q]} for r in requests]
        rows = score(engine, reqs, {f"{r['id']}:{q['id']}": gold[f"{r['id']}:passed"] for r in requests})
        if spec.get("inverted"):
            rows = [dict(r, probs=r["probs"][::-1]) for r in rows]
        out[q["id"]] = {"accuracy": metrics(rows)["accuracy"], **gate_errors(rows)}
        keep(f"{split}/paraphrase:{q['id']}", rows)
    return out


def breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    return {k: {"n": len(v), "accuracy": metrics(v)["accuracy"]} for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--temperatures")
    ap.add_argument("--splits", nargs="*", default=["dev"])
    ap.add_argument("--typed", nargs="*", default=[])
    ap.add_argument("--paraphrase", action="store_true", help="also ask the gate in the PARAPHRASES wordings")
    ap.add_argument("--final-paraphrase", action="store_true", help="also ask the held-out FINAL_PARAPHRASES (test, ood)")
    ap.add_argument("--ecosystems", action="store_true", help="also score ecosystem_probes.py (held out; final stage)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, help="also write every scored question as JSON lines (for paired bootstraps)")
    args = ap.parse_args()
    from jev_service.engine import DecisionEngine
    engine = DecisionEngine(args.checkpoint, args.model_path, temperatures=args.temperatures)
    report = {"checkpoint": args.checkpoint, "sha256": engine.checkpoint_sha256, "splits": {}, "typed": {}}
    predictions = []

    def keep(label, rows):
        predictions.extend({"set": label, "id": r["id"], "case_id": r["case_id"], "type": r["type"],
                            "target": r["target"], "probs": r["probs"]} for r in rows)

    for split in args.splits:
        reqs = [json.loads(l) for l in open(HERE / "data" / f"{split}_requests.jsonl", encoding="utf-8")]
        gold = {g["id"]: g for g in map(json.loads, open(HERE / "data" / f"{split}_questions.jsonl", encoding="utf-8"))}
        rows = score(engine, reqs, gold)
        keep(split, rows)
        report["splits"][split] = {"overall": metrics(rows), "gate": gate_errors(rows),
                                   "by_type": {t: metrics([r for r in rows if r["type"] == t]) for t in ("boolean", "choice", "score")},
                                   "by_runner": breakdown(rows, "runner"), "by_outcome": breakdown(rows, "outcome"),
                                   "by_exit_code_shown": breakdown([r for r in rows if r["workflow"] == "test_runs"], "exit_code_shown")}
        print(f"{split}: acc {report['splits'][split]['overall']['accuracy']:.4f} | gate false-pass "
              f"{report['splits'][split]['gate']['false_pass']}/{report['splits'][split]['gate']['n']} | "
              + " ".join(f"{t} {m['accuracy']:.3f}" for t, m in report["splits"][split]["by_type"].items() if m), flush=True)
        if args.paraphrase:
            report["splits"][split]["paraphrase"] = para = paraphrased(engine, reqs, gold, keep, split)
            print(f"{split} paraphrased gate:", " ".join(f"{k} {v['accuracy']:.3f} (false-pass {v['false_pass']})"
                                                       for k, v in para.items()), flush=True)
        if args.final_paraphrase:
            if split not in ("test", "ood"):
                raise SystemExit("FINAL_PARAPHRASES are scored on test and ood only")
            report["splits"][split]["final_paraphrase"] = para = paraphrased(engine, reqs, gold, keep, split, FINAL_PARAPHRASES)
            print(f"{split} held-out wordings:", " ".join(f"{k} {v['accuracy']:.3f} (false-pass {v['false_pass']})"
                                                         for k, v in para.items()), flush=True)
    for split in args.typed:
        reqs = [json.loads(l) for l in open(ROOT / "typed_decisions" / "prepared" / f"{split}_requests.jsonl", encoding="utf-8")]
        gold = {}
        for g in map(json.loads, open(ROOT / "typed_decisions" / "prepared" / f"{split}_questions.jsonl", encoding="utf-8")):
            gold[g["id"]] = {"id": g["id"], "case_id": g["case_id"], "type": g["question"]["type"],
                             "workflow": g["workflow"], "target": g["target"]}
        rows = score(engine, reqs, gold)
        keep(f"typed_{split}", rows)
        report["typed"][split] = metrics(rows)
        print(f"typed {split}: acc {report['typed'][split]['accuracy']:.4f} soft CE {report['typed'][split]['soft_cross_entropy']:.4f}", flush=True)
    probes = engine.evaluate({"requests": [{"id": str(i), "state": s, "questions": [PROBE_Q]} for i, (s, _) in enumerate(PROBES)]})
    report["probes"] = [{"state": s, "gold_true": g, "p_true": r["answers"][0]["probability"]}
                        for (s, g), r in zip(PROBES, probes["results"])]
    wrong = [p for p in report["probes"] if (p["p_true"] >= .5) != p["gold_true"]]
    print(f"probes wrong: {len(wrong)}/{len(PROBES)}", [f"{p['state'][:32]}… {p['p_true']:.2f}" for p in wrong], flush=True)
    if args.ecosystems:
        from coding_gates.build_dataset import GATE
        from coding_gates.ecosystem_probes import PROBES as ECOSYSTEMS, state
        res = engine.evaluate({"requests": [{"id": name, "state": state(command, output), "questions": [GATE]}
                                            for name, _, command, output in ECOSYSTEMS]})
        report["ecosystems"] = [{"id": name, "gold_true": gold, "p_true": r["answers"][0]["probability"]}
                                for (name, gold, _, _), r in zip(ECOSYSTEMS, res["results"])]
        wrong = [e for e in report["ecosystems"] if (e["p_true"] >= .5) != e["gold_true"]]
        print(f"ecosystems wrong: {len(wrong)}/{len(ECOSYSTEMS)}", [f"{e['id']} {e['p_true']:.2f}" for e in wrong], flush=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.predictions:
        args.predictions.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
