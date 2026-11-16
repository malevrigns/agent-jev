"""Coding gates: labels follow the planted outcome, and recorded output names no machine."""
import gzip
import json
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coding_gates import build_dataset, evaluate, make_runs, wordings  # noqa: E402
from jev_service.contract import prepare  # noqa: E402

GATES = ROOT / "coding_gates"


class Sanitize(unittest.TestCase):
    def setUp(self):
        self.saved = make_runs.JS, make_runs.WORK, make_runs.PY_PREFIX
        make_runs.JS = Path("C:/Users/someone/tools")
        make_runs.WORK = Path("C:/Users/someone/scratch")
        make_runs.PY_PREFIX = Path("C:/Users/someone/venvs/pytest")

    def tearDown(self):
        make_runs.JS, make_runs.WORK, make_runs.PY_PREFIX = self.saved

    def test_every_separator_style(self):
        root = make_runs.WORK / "py" / "r00007"
        # plain, Windows, JSON-escaped, and escaped twice (TAP's requireStack)
        for sep in ("/", "\\", "\\\\", "\\\\\\\\"):
            printed = [path.as_posix().replace("/", sep) for path in (
                root / "test_x.py", make_runs.JS / "node_modules", make_runs.PY_PREFIX / "Lib",
                Path("C:/Users/someone/elsewhere"))]
            expected = [path.replace("/", sep) for path in (
                "C:/dev/cart-api/test_x.py", "C:/dev/tools/node_modules", "C:/dev/python/Lib", "C:/Users/dev/elsewhere")]
            self.assertEqual(make_runs.sanitize("\n".join(printed), root, "cart-api").split("\n"), expected, repr(sep))

    def test_a_short_python_prefix_is_never_aliased(self):
        make_runs.PY_PREFIX = Path("C:/")
        self.assertEqual(make_runs.sanitize("C:/Windows/System32", make_runs.WORK / "r1", "x"), "C:/Windows/System32")


class Labels(unittest.TestCase):
    def test_outcome_answers(self):
        expected = {"pass": (0, "commit", True), "fail": (1, "fix_assertion", True),
                    "error": (1, "fix_exception", True), "broken": (1, "fix_load", False),
                    "no_tests": (1, "find_tests", False)}
        for outcome, (gate, action, counted) in expected.items():
            answers = {q["id"]: target for q, target in build_dataset.questions_for(outcome, 0 if outcome == "pass" else 2)}
            self.assertEqual(answers["passed"].index(1.0), gate)
            self.assertEqual(build_dataset.NEXT_KEYS[answers["next"].index(1.0)], action)
            self.assertEqual("failures" in answers, counted)

    def test_count_levels(self):
        self.assertEqual([build_dataset.count_level(n) for n in (0, 1, 2, 5, 6, 40)], [0, 1, 2, 2, 3, 3])

    def test_summaries_state_their_counts(self):
        for case in build_dataset.summary_cases(random.Random(0), 400):
            self.assertEqual(case["outcome"] == "pass", case["n_failed"] == 0)
            if case["n_failed"] > 1:
                self.assertIn(str(case["n_failed"]), case["state"])


class Wordings(unittest.TestCase):
    def test_no_evaluation_wording_is_trained_on(self):
        held_out, trained = set(), set()
        for q in evaluate.PARAPHRASES + evaluate.FINAL_PARAPHRASES + [evaluate.PROBE_Q]:
            held_out.add(q["question"]); held_out.update(q.get("criteria", {}).values())
        for text, criteria, _ in wordings.GATE:
            trained.add(text); trained.update((criteria or {}).values())
        self.assertEqual(held_out & trained, set())

    def test_reworded_requests_keep_their_answers(self):
        rng = random.Random(0)
        inverted = {text for text, _, flips in wordings.GATE if flips}
        pairs = build_dataset.questions_for("fail", 1)
        request = {"id": "r", "state": {"command": "pytest -q", "output": "1 failed, 3 passed", "exit_code": 1},
                   "questions": [q for q, _ in pairs]}
        targets = [t for _, t in pairs]
        for _ in range(300):
            new, new_targets = wordings.vary(request, targets, rng)
            prepare(new)  # still a valid API request
            gate = new["questions"][0]["question"]
            self.assertEqual(new_targets[0], targets[0][::-1] if gate in inverted else targets[0])
            self.assertEqual(new_targets[1:], targets[1:])
            self.assertEqual(list(new["questions"][1]["options"]), build_dataset.NEXT_KEYS)
            self.assertIn("1 failed, 3 passed", json.dumps(new["state"]))
        self.assertEqual(request["questions"][0]["question"], build_dataset.GATE["question"])


class EcosystemProbes(unittest.TestCase):
    def test_probes_are_valid_requests_of_both_kinds(self):
        from coding_gates.ecosystem_probes import PROBES, state
        self.assertEqual(len({name for name, *_ in PROBES}), len(PROBES))
        self.assertEqual({gold for _, gold, *_ in PROBES}, {True, False})
        prepare({"requests": [{"id": name, "state": state(command, output), "questions": [build_dataset.GATE]}
                              for name, _, command, output in PROBES]})


class RecordedRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [json.loads(line) for line in gzip.open(GATES / "runs.jsonl.gz", "rt", encoding="utf-8")]

    def test_every_run_matched_its_planted_outcome(self):
        self.assertEqual(len(self.runs), 2100)
        for r in self.runs:
            self.assertTrue(r["consistent"], r["id"])
            green_exit = r["outcome"] == "pass" or (r["outcome"] == "no_tests" and r["runner"] in ("mocha", "node"))
            self.assertEqual(r["exit_code"] == 0, green_exit, r["id"])
            self.assertEqual(r["n_failed"] > 0, r["outcome"] in ("fail", "error"), r["id"])

    def test_no_machine_paths(self):
        users = re.compile(r"C:(\\+|/)Users(\\+|/)(?!dev\b)")
        self.assertEqual([r["id"] for r in self.runs if users.search(r["output"])], [])

    def test_manifest_covers_every_run_and_summary(self):
        manifest = json.loads((GATES / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(s["requests"] for s in manifest["splits"].values()), len(self.runs) + 1800)
        held_out = {r["id"] for r in self.runs if r["runner"] in ("vitest", "node")}
        self.assertEqual(manifest["splits"]["ood"]["requests"], len(held_out))


if __name__ == "__main__":
    unittest.main()
