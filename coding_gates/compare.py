"""Paired, case-level bootstrap of the accuracy difference between two evaluate.py --predictions files.

usage: python coding_gates/compare.py BASELINE_pred.jsonl[.gz] CANDIDATE_pred.jsonl[.gz] [--out compare.json]
Every set (dev, test, ood, each paraphrase, typed_test, ...) is compared on the questions both files
scored. Cases are resampled with replacement, keeping all of a case's questions together.
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    sets = defaultdict(dict)
    for line in (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8"):
        p = json.loads(line)
        sets[p["set"]][p["id"]] = p
    return sets


def correct(p):
    return float(np.argmax(p["probs"]) == np.argmax(p["target"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    a, b = load(args.baseline), load(args.candidate)
    rng = np.random.default_rng(args.seed)
    report = {}
    for name in sorted(set(a) & set(b)):
        ids = sorted(set(a[name]) & set(b[name]))
        cases = sorted({a[name][i]["case_id"] for i in ids})
        index = {c: k for k, c in enumerate(cases)}
        hits_a, hits_b, count = np.zeros(len(cases)), np.zeros(len(cases)), np.zeros(len(cases))
        for i in ids:
            k = index[a[name][i]["case_id"]]
            hits_a[k] += correct(a[name][i]); hits_b[k] += correct(b[name][i]); count[k] += 1
        picks = rng.integers(0, len(cases), (args.samples, len(cases)))
        diffs = (hits_b[picks].sum(1) - hits_a[picks].sum(1)) / count[picks].sum(1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        report[name] = {"questions": len(ids), "cases": len(cases),
                        "baseline": hits_a.sum() / count.sum(), "candidate": hits_b.sum() / count.sum(),
                        "difference": (hits_b.sum() - hits_a.sum()) / count.sum(), "ci95": [lo, hi]}
        r = report[name]
        print(f"{name:24s} {r['baseline']:.4f} -> {r['candidate']:.4f}  {r['difference']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  ({len(ids)} questions, {len(cases)} cases)")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
