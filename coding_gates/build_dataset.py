"""Turn recorded test runs and terse summaries into AgentJev requests with exact answers.

Splits (decided here, before any model is scored):
  train / dev / test   real runs from pytest, unittest, jest, mocha (70/15/15 per runner x outcome)
                       and one-line summaries (70/15/15)
  ood                  real runs from vitest and node --test: never trained on or used for selection
Every request is one state with its questions. Each question carries `target`, a one-hot list in
candidate order, and `case_id`, `workflow`, `type`, so the benchmark scripts can score it.
Outputs that would encode past 2,000 tokens keep their tail, where runners print the summary.

usage: python coding_gates/build_dataset.py [--model-path models/agentjev-v1]
The files land in coding_gates/data/ and are checked against coding_gates/manifest.json.
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "data"
SEED = 20261014
MAX_PATH_TOKENS = 2000

GATE = {"id": "passed", "type": "boolean", "question": "Have all tests passed?",
        "criteria": {"true": "At least one test ran and every test that ran passed.",
                     "false": "A test failed or raised an error, the tests could not load, or no tests ran."}}
NEXT_KEYS = ["commit", "fix_assertion", "fix_exception", "fix_load", "find_tests"]
NEXT = {"id": "next", "type": "choice", "question": "What should the coding agent do next?",
        "options": {"commit": "The suite is green; commit the change.",
                    "fix_assertion": "A test assertion failed; read the expected and actual values and fix the code.",
                    "fix_exception": "The code under test raised an exception; fix the error it reports.",
                    "fix_load": "The tests could not load because of a syntax or import error; fix that first.",
                    "find_tests": "No tests were found; check the test file names or add tests."}}
OUTCOME_NEXT = {"pass": "commit", "fail": "fix_assertion", "error": "fix_exception", "broken": "fix_load",
                "no_tests": "find_tests"}
COUNT_LEVELS = ["No test failed or errored.", "Exactly one test failed or errored.",
                "Two to five tests failed or errored.", "Six or more tests failed or errored."]
COUNT = {"id": "failures", "type": "score", "question": "How many tests failed or raised an error?",
         "levels": COUNT_LEVELS}


def count_level(n):
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 5 else 3


def one_hot(n, i):
    return [1.0 if j == i else 0.0 for j in range(n)]


def questions_for(outcome, n_bad):
    qs = [(GATE, one_hot(2, 0 if outcome == "pass" else 1)),
          (NEXT, one_hot(len(NEXT_KEYS), NEXT_KEYS.index(OUTCOME_NEXT[outcome])))]
    if outcome in ("pass", "fail", "error"):
        qs.append((COUNT, one_hot(4, count_level(n_bad))))
    return qs


def trim(output, budget):
    if len(output) <= budget:
        return output
    return "…[earlier output truncated]…\n" + output[-budget:]


def summary_cases(rng, n):
    """One-line results in the shapes agents and CI logs actually use; counts are exact."""
    rows = []
    for i in range(n):
        total = rng.randint(1, 400)
        outcome = rng.choices(["pass", "fail", "error"], weights=[40, 40, 20])[0]
        bad = 0 if outcome == "pass" else rng.choice([1, 1, 1, 2, 3, 4, 7, 12, rng.randint(1, max(1, total // 3))])
        bad = min(bad, total)
        ok = total - bad
        skipped = rng.choice([0, 0, 0, rng.randint(1, 5)])
        secs = f"{rng.uniform(0.05, 90):.2f}"
        if outcome == "error":
            text = rng.choice([
                f"Ran {total} tests in {secs}s\n\nFAILED (errors={bad})",
                f"{ok} passed, {bad} error{'s' if bad > 1 else ''} in {secs}s",
                f"Tests run: {total}, Failures: 0, Errors: {bad}, Skipped: {skipped}",
                f"测试 {total} 个，通过 {ok} 个，{bad} 个报错（异常）",
                f"{bad} of {total} tests raised an exception; {ok} passed.",
            ])
        elif outcome == "fail":
            text = rng.choice([
                f"{ok} tests passed, {bad} failed.",
                f"{ok} passed, {bad} failed in {secs}s",
                f"Tests run: {total}, Failures: {bad}, Errors: 0, Skipped: {skipped}",
                f"Ran {total} tests in {secs}s\n\nFAILED (failures={bad})",
                f"Tests:       {bad} failed, {ok} passed, {total} total",
                f"  {ok} passing ({secs}s)\n  {bad} failing",
                f"测试结果：共 {total} 个，通过 {ok} 个，失败 {bad} 个",
                f"{bad}/{total} tests failing on the current patch",
                f"All tests passed except {bad}." if bad > 1 else "All tests passed except one.",
            ])
        else:
            text = rng.choice([
                f"{total} tests passed.",
                f"{total} passed in {secs}s",
                f"Tests run: {total}, Failures: 0, Errors: 0, Skipped: {skipped}",
                f"Ran {total} tests in {secs}s\n\nOK" + (f" (skipped={skipped})" if skipped else ""),
                f"Tests:       {total} passed, {total} total",
                f"  {total} passing ({secs}s)",
                f"测试全部通过：{total} 个",
                f"All {total} tests passed; 0 failures.",
                f"{total} passed, 0 failed, {skipped} skipped" if skipped else f"{total} passed, 0 failed",
            ])
        rows.append({"id": f"summary-{i:05d}", "workflow": "summary", "outcome": outcome, "n_failed": bad,
                     "state": text, "runner": "summary"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=ROOT / "models" / "agentjev-v1", help="folder with the tokenizer")
    ap.add_argument("--runs", type=Path, default=HERE / "runs.jsonl.gz")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer
    from jev_service.contract import encode_paths, prepare
    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    rng = random.Random(SEED)
    runs = [json.loads(line) for line in gzip.open(args.runs, "rt", encoding="utf-8")]
    runs = [r for r in runs if r["consistent"]]
    cases = []
    for r in runs:
        show_exit = rng.random() < .5
        state = {"command": r["command"], "output": r["output"]}
        if show_exit:
            state["exit_code"] = r["exit_code"]
        cases.append({"id": r["id"], "workflow": "test_runs", "runner": r["runner"], "outcome": r["outcome"],
                      "n_failed": r["n_failed"], "state": state, "exit_code_shown": show_exit})
    cases += summary_cases(rng, 1800)

    splits = defaultdict(list)
    groups = defaultdict(list)
    for c in cases:
        groups[(c["runner"], c["outcome"])].append(c)
    for (runner, _), group in sorted(groups.items()):
        rng.shuffle(group)
        if runner in ("vitest", "node"):
            splits["ood"] += group
            continue
        a, b = int(len(group) * .70), int(len(group) * .85)
        splits["train"] += group[:a]; splits["dev"] += group[a:b]; splits["test"] += group[b:]

    OUT.mkdir(exist_ok=True)
    stats = {}
    for split, items in splits.items():
        rng.shuffle(items)
        req_rows, q_rows, dropped = [], [], 0
        for c in items:
            qs = questions_for(c["outcome"], c["n_failed"])
            state = c["state"]
            for budget in (6000, 4000, 3000, 2200, 1600, 1100):
                if isinstance(state, dict):
                    state = dict(c["state"], output=trim(c["state"]["output"], budget))
                request = {"id": c["id"], "state": state, "questions": [q for q, _ in qs]}
                paths = encode_paths(prepare(request), tok, 1 << 20)[0]
                if max(map(len, paths)) <= MAX_PATH_TOKENS:
                    break
            else:
                dropped += 1
                continue
            req_rows.append(request)
            for q, target in qs:
                q_rows.append({"id": f"{c['id']}:{q['id']}", "case_id": c["id"], "workflow": c["workflow"],
                               "runner": c["runner"], "outcome": c["outcome"], "type": q["type"],
                               "exit_code_shown": c.get("exit_code_shown"), "target": target})
        hashes = {}
        for kind, rows in (("requests", req_rows), ("questions", q_rows)):
            blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
            (OUT / f"{split}_{kind}.jsonl").write_bytes(blob)
            hashes[f"{kind}_sha256"] = hashlib.sha256(blob).hexdigest()
        stats[split] = {"requests": len(req_rows), "questions": len(q_rows), "dropped_too_long": dropped, **hashes}
    manifest = {"seed": SEED, "max_path_tokens": MAX_PATH_TOKENS, "splits": dict(sorted(stats.items()))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {"requests": v["requests"], "questions": v["questions"]} for k, v in stats.items()}))
    published = HERE / "manifest.json"
    if published.exists():
        same = json.loads(published.read_text(encoding="utf-8")) == manifest
        print("matches coding_gates/manifest.json" if same else "DIFFERS from coding_gates/manifest.json")


if __name__ == "__main__":
    main()
