"""Generate small projects with a known test outcome, run real test runners, record what they print.

The outcome is built into each project, never read back from the output:
  pass      every test passes (sometimes one or two are skipped)
  fail      k tests fail an assertion
  error     k tests hit an exception raised by the code under test
  broken    the module cannot be parsed or imported, so no test runs
  no_tests  the runner finds no tests
Every run's exit code is checked against its outcome and the run is dropped
on a mismatch. Expected values come from the Python implementation; the JS
twin of each function must agree, or its "pass" projects fail and drop out.

The recorded set, coding_gates/runs.jsonl.gz, came from Windows 11 with Python 3.12.14 and
pytest 8.4.2, Node 24.18.0, jest 30.5.2, mocha 12.0.2 and vitest 5.0.2:

    npm install --prefix TOOLS jest@30.5.2 mocha@12.0.2 vitest@5.0.2
    python coding_gates/make_runs.py runs.jsonl --js-tools TOOLS --python PYTHON_WITH_PYTEST

Other versions print slightly different text; the outcomes and labels stay the same.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set by main(): the Python that has pytest, the folder whose node_modules has the JS runners
# (JS projects are written under it), and the scratch folder for Python projects.
PY = Path(sys.executable)
JS = Path("js-tools")
WORK = Path(tempfile.gettempdir()) / "agentjev_coding_gates"
PY_PREFIX = None  # sys.prefix of PY; its paths show up in tracebacks
TRAIN_RUNNERS = ("pytest", "unittest", "jest", "mocha")
OOD_RUNNERS = ("vitest", "node")  # never used for training or model selection

# name: (python source, js source, argument sampler, python implementation)
WORDS = "alpha beta gamma delta token cart order invoice user report queue cache build test value".split()
FUNCS = {
    "add": ("def add(a, b):\n    return a + b\n", "function add(a, b) {\n  return a + b;\n}\n",
            lambda r: (r.randint(-50, 99), r.randint(-50, 99)), lambda a, b: a + b),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              "function clamp(x, lo, hi) {\n  return Math.max(lo, Math.min(x, hi));\n}\n",
              lambda r: (r.randint(-20, 120), 0, 100), lambda x, lo, hi: max(lo, min(x, hi))),
    "is_even": ("def is_even(n):\n    return n % 2 == 0\n", "function is_even(n) {\n  return n % 2 === 0;\n}\n",
                lambda r: (r.randint(-30, 99),), lambda n: n % 2 == 0),
    "factorial": ("def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n",
                  "function factorial(n) {\n  let out = 1;\n  for (let i = 2; i <= n; i++) out *= i;\n  return out;\n}\n",
                  lambda r: (r.randint(0, 12),), lambda n: __import__("math").factorial(n)),
    "fib": ("def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
            "function fib(n) {\n  let a = 0, b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}\n",
            lambda r: (r.randint(0, 25),), None),
    "sum_list": ("def sum_list(xs):\n    return sum(xs)\n", "function sum_list(xs) {\n  return xs.reduce((s, x) => s + x, 0);\n}\n",
                 lambda r: ([r.randint(-9, 40) for _ in range(r.randint(0, 6))],), lambda xs: sum(xs)),
    "max_of": ("def max_of(xs):\n    return max(xs)\n", "function max_of(xs) {\n  return Math.max(...xs);\n}\n",
               lambda r: ([r.randint(-9, 99) for _ in range(r.randint(1, 6))],), lambda xs: max(xs)),
    "discount": ("def discount(cents, pct):\n    return cents * (100 - pct) // 100\n",
                 "function discount(cents, pct) {\n  return Math.floor(cents * (100 - pct) / 100);\n}\n",
                 lambda r: (r.randint(100, 99999), r.choice([0, 5, 10, 15, 20, 25, 50])),
                 lambda c, p: c * (100 - p) // 100),
    "reverse": ("def reverse(s):\n    return s[::-1]\n", "function reverse(s) {\n  return s.split('').reverse().join('');\n}\n",
                lambda r: ("".join(r.choice("abcdefghij") for _ in range(r.randint(1, 9))),), lambda s: s[::-1]),
    "count_vowels": ("def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())\n",
                     "function count_vowels(s) {\n  return [...s.toLowerCase()].filter((c) => 'aeiou'.includes(c)).length;\n}\n",
                     lambda r: (" ".join(r.sample(WORDS, r.randint(1, 3))),), lambda s: sum(c in "aeiou" for c in s.lower())),
    "slugify": ("def slugify(s):\n    return '-'.join(s.lower().split())\n",
                "function slugify(s) {\n  return s.toLowerCase().trim().split(/\\s+/).join('-');\n}\n",
                lambda r: (" ".join(w.capitalize() if r.random() < .5 else w for w in r.sample(WORDS, r.randint(1, 4))),),
                lambda s: "-".join(s.lower().split())),
    "word_count": ("def word_count(s):\n    return len(s.split())\n", "function word_count(s) {\n  return s.trim().split(/\\s+/).length;\n}\n",
                   lambda r: (" ".join(r.sample(WORDS, r.randint(1, 6))),), lambda s: len(s.split())),
    "title_case": ("def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
                   "function title_case(s) {\n  return s.split(/\\s+/).map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase()).join(' ');\n}\n",
                   lambda r: (" ".join(r.sample(WORDS, r.randint(1, 4))),), lambda s: " ".join(w.capitalize() for w in s.split())),
    "pad_left": ("def pad_left(s, n):\n    return s.rjust(n, '0')\n", "function pad_left(s, n) {\n  return s.padStart(n, '0');\n}\n",
                 lambda r: (str(r.randint(0, 999)), r.randint(1, 6)), lambda s, n: s.rjust(n, "0")),
    "is_palindrome": ("def is_palindrome(s):\n    return s == s[::-1]\n",
                      "function is_palindrome(s) {\n  return s === s.split('').reverse().join('');\n}\n",
                      lambda r: (r.choice(["level", "rotor", "civic", "abba", "token", "cache", "noon", "build"]),),
                      lambda s: s == s[::-1]),
}


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


FUNCS["fib"] = FUNCS["fib"][:3] + (_fib,)
THEMES = {
    "mathutil": ["add", "clamp", "is_even", "factorial", "fib", "sum_list", "max_of"],
    "textutil": ["reverse", "count_vowels", "slugify", "word_count", "title_case", "pad_left", "is_palindrome"],
    "pricing": ["discount", "clamp", "sum_list", "add", "max_of"],
    "strings": ["slugify", "title_case", "pad_left", "reverse", "word_count"],
}
PROJECTS = ["billing-service", "text-tools", "cart-api", "invoice-worker", "report-builder", "auth-gateway",
            "queue-consumer", "pricing-engine", "user-profile", "notify-hub", "search-index", "export-cli"]
MISSING = {"py": ["fastjsonschema_ext", "yaml_tools", "money_utils", "slugger"],
           "js": ["left-pad-extended", "string-slugger", "money-utils", "lodash-plus"]}


def lit(value, lang):
    if isinstance(value, bool):
        return ("True" if value else "False") if lang == "py" else ("true" if value else "false")
    return json.dumps(value)


def perturb(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + random.Random(value).choice([1, -1, 2, 10])
    return value + "x" if value else "x"


def plan(index, runner, rng):
    """Choose the module, the tests and the outcome for one project."""
    theme = rng.choice(list(THEMES))
    funcs = THEMES[theme]
    outcome = rng.choices(["pass", "fail", "error", "broken", "no_tests"], weights=[35, 30, 15, 12, 8])[0]
    n = rng.randint(2, 24)
    tests = []
    for t in range(n):
        fn = rng.choice(funcs)
        args = FUNCS[fn][2](rng)
        tests.append({"fn": fn, "args": args, "expected": FUNCS[fn][3](*args),
                      "name": f"{fn}_{rng.choice(['basic', 'edge', 'case', 'sample', 'regression'])}_{t + 1}"})
    failed, skipped = [], []
    broken_fn = None
    if outcome == "fail":
        failed = rng.sample(range(n), rng.randint(1, min(6, n)))
        for i in failed:
            tests[i]["expected"] = perturb(tests[i]["expected"])
    elif outcome == "error":
        used = sorted({t["fn"] for t in tests})
        broken_fn = rng.choice(used)
        failed = [i for i, t in enumerate(tests) if t["fn"] == broken_fn]
    elif outcome == "pass" and rng.random() < .3 and n > 3:
        skipped = rng.sample(range(n), rng.randint(1, 2))
    return {"index": index, "runner": runner, "theme": theme, "funcs": funcs, "outcome": outcome, "tests": tests,
            "failed": sorted(failed), "skipped": sorted(skipped), "broken_fn": broken_fn,
            "broken_kind": rng.choice(["syntax", "import"]) if outcome == "broken" else None,
            "project": rng.choice(PROJECTS), "flags": rng.random()}


def module_source(p, lang):
    parts = []
    if p["broken_kind"] == "import":
        dep = random.Random(p["index"]).choice(MISSING[lang])
        parts.append(f"import {dep}\n" if lang == "py" else f"const extra = require('{dep}');\n")
    for fn in p["funcs"]:
        src = FUNCS[fn][0 if lang == "py" else 1]
        if fn == p["broken_fn"]:
            # The code under test calls a helper that was never written.
            if lang == "py":
                src = src.replace("    return ", "    return normalize_input(", 1).rstrip("\n") + ")\n"
            else:
                src = src.replace("  return ", "  return normalizeInput(", 1).replace(";\n}", ");\n}", 1)
        parts.append(src)
    if p["broken_kind"] == "syntax":
        victim = p["funcs"][random.Random(p["index"]).randrange(len(p["funcs"]))]
        idx = next(i for i, s in enumerate(parts) if f"{victim}(" in s.split("\n")[0])
        parts[idx] = parts[idx].replace("):\n", ")\n", 1) if lang == "py" else parts[idx].replace("\n}\n", "\n\n", 1)
    return "\n\n".join(parts)


def write_project(p, root):
    root.mkdir(parents=True, exist_ok=True)
    mod, runner, tests = p["theme"], p["runner"], p["tests"]
    no_tests = p["outcome"] == "no_tests"
    if runner in ("pytest", "unittest"):
        (root / f"{mod}.py").write_text(module_source(p, "py"), encoding="utf-8")
        names = ", ".join(sorted({t["fn"] for t in tests}))
        if runner == "pytest":
            lines = ["import pytest", f"from {mod} import {names}", ""]
            for i, t in enumerate(tests):
                prefix = "check" if no_tests else "test"
                if i in p["skipped"]:
                    lines.append(f"@pytest.mark.skip(reason=\"flaky on CI, see issue #{100 + i}\")")
                call = f"{t['fn']}({', '.join(lit(a, 'py') for a in t['args'])})"
                lines += [f"def {prefix}_{t['name']}():", f"    assert {call} == {lit(t['expected'], 'py')}", ""]
            (root / f"test_{mod}.py").write_text("\n".join(lines), encoding="utf-8")
        else:
            lines = ["import unittest", f"from {mod} import {names}", "", f"class {mod.capitalize()}Tests(unittest.TestCase):"]
            for i, t in enumerate(tests):
                prefix = "check" if no_tests else "test"
                if i in p["skipped"]:
                    lines.append(f"    @unittest.skip(\"flaky on CI, see issue #{100 + i}\")")
                call = f"{t['fn']}({', '.join(lit(a, 'py') for a in t['args'])})"
                lines += [f"    def {prefix}_{t['name']}(self):", f"        self.assertEqual({call}, {lit(t['expected'], 'py')})", ""]
            lines += ["", "if __name__ == '__main__':", "    unittest.main()"]
            (root / f"test_{mod}.py").write_text("\n".join(lines), encoding="utf-8")
        return
    esm = runner == "vitest"
    ext = ".mjs" if esm else ".js"
    src = module_source(p, "js")
    exports = ", ".join(p["funcs"])
    src = (src.replace("const extra = require(", "import extra from ").replace("');\n", "';\n", 1)
           if esm and p["broken_kind"] == "import" else src)
    if esm and p["broken_kind"] == "import":
        src = re.sub(r"import extra from \('([^']+)'\);", r"import extra from '\1';", src)
    src += f"\n{'export { ' + exports + ' };' if esm else 'module.exports = { ' + exports + ' };'}\n"
    (root / "src").mkdir(exist_ok=True)
    (root / "test").mkdir(exist_ok=True)
    (root / "src" / f"{mod}{ext}").write_text(src, encoding="utf-8")
    names = ", ".join(sorted({t["fn"] for t in tests}))
    head = {"jest": [f"const {{ {names} }} = require('../src/{mod}');"],
            "mocha": ["const assert = require('node:assert/strict');", f"const {{ {names} }} = require('../src/{mod}');"],
            "node": ["const test = require('node:test');", "const assert = require('node:assert/strict');",
                     f"const {{ {names} }} = require('../src/{mod}');"],
            "vitest": ["import { test, expect } from 'vitest';", f"import {{ {names} }} from '../src/{mod}.mjs';"]}[runner]
    body = []
    for i, t in enumerate(tests):
        call = f"{t['fn']}({', '.join(lit(a, 'js') for a in t['args'])})"
        exp = lit(t["expected"], "js")
        title = t["name"].replace("_", " ")
        skip = i in p["skipped"]
        if runner in ("jest", "vitest"):
            body.append(f"test{'.skip' if skip else ''}('{title}', () => {{\n  expect({call}).toBe({exp});\n}});")
        elif runner == "mocha":
            body.append(f"  it{'.skip' if skip else ''}('{title}', () => {{\n    assert.strictEqual({call}, {exp});\n  }});")
        else:
            body.append(f"test('{title}', {{ skip: {'true' if skip else 'false'} }}, () => {{\n  assert.strictEqual({call}, {exp});\n}});")
    if no_tests:
        body = [f"// TODO: port the {mod} checks from the old suite"]
    if runner == "mocha" and not no_tests:
        body = [f"describe('{mod}', () => {{"] + body + ["});"]
    name = f"{mod}.checks{ext}" if (no_tests and runner in ("jest", "vitest")) else f"{mod}.test{ext}"
    (root / "test" / name).write_text("\n".join(head + [""] + body) + "\n", encoding="utf-8")


def command(p, root):
    r, f = p["runner"], p["flags"]
    mod = p["theme"]
    if r == "pytest":
        flags = ["-q", "", "-v", "-q --tb=short", "-v --tb=line", "-rA -q", "--tb=long"][int(f * 7)]
        shown = f"pytest {flags}".strip()
        return [str(PY), "-m", "pytest", "-p", "no:cacheprovider", *flags.split()], root, shown
    if r == "unittest":
        flags = ["", "-v"][int(f * 2)]
        shown = f"python -m unittest {flags} test_{mod}".replace("  ", " ")
        return [str(PY), "-m", "unittest", *flags.split(), f"test_{mod}"], root, shown
    rel = root.relative_to(JS).as_posix()
    if r == "jest":
        flags = ["", "--verbose"][int(f * 2)]
        return ([ "node", "node_modules/jest/bin/jest.js", "--rootDir", rel, "--ci", "--maxWorkers=1", *flags.split()],
                JS, f"npx jest {flags}".strip())
    if r == "mocha":
        rep = ["spec", "dot"][int(f * 2)]
        return (["node", "node_modules/mocha/bin/mocha.js", "--reporter", rep, f"{rel}/test"], JS,
                f"npx mocha --reporter {rep} test/" if rep != "spec" else "npx mocha test/")
    if r == "vitest":
        rep = ["default", "verbose", "dot"][int(f * 3)]
        return (["node", "node_modules/vitest/vitest.mjs", "run", "--root", rel, f"--reporter={rep}", "--maxWorkers=1"],
                JS, "npx vitest run" + ("" if rep == "default" else f" --reporter={rep}"))
    rep = ["spec", "tap", "dot"][int(f * 3)]
    files = sorted(f"{rel}/test/{t.name}" for t in (root / "test").iterdir())
    shown = [f"test/{Path(x).name}" for x in files]
    return (["node", "--test", f"--test-reporter={rep}", *files], JS,
            " ".join(["node --test"] + ([] if rep == "spec" else [f"--test-reporter={rep}"]) + shown))


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def sanitize(text, root, project):
    text = ANSI.sub("", text).replace("\r\n", "\n")
    aliases = [(root, f"C:/dev/{project}"), (JS, "C:/dev/tools"), (WORK, "C:/dev/tools")]
    if PY_PREFIX is not None and len(PY_PREFIX.parts) > 2:  # never alias a bare drive or top folder
        aliases.append((PY_PREFIX, "C:/dev/python"))
    for path, alias in aliases:
        # The same path may be printed with /, \ or JSON-escaped \\ separators.
        for sep in ("\\\\\\\\", "\\\\", "\\", "/"):
            text = text.replace(path.as_posix().replace("/", sep), alias.replace("/", sep))
    text = re.sub(r"C:(\\+|/)Users(\\+|/)[A-Za-z0-9_.-]+", lambda m: f"C:{m.group(1)}Users{m.group(2)}dev", text)
    return text.strip("\n")


EXPECTED_EXIT = {"pass": {0}, "fail": None, "error": None, "broken": None,
                 "no_tests": {"pytest": {5}, "unittest": {5}, "jest": {1}, "mocha": {0}, "vitest": {1}, "node": {0}}}


def run_one(p):
    root = (WORK / "py" / f"r{p['index']:05d}") if p["runner"] in ("pytest", "unittest") else (JS / "projects" / f"r{p['index']:05d}")
    shutil.rmtree(root, ignore_errors=True)
    write_project(p, root)
    argv, cwd, shown = command(p, root)
    env = dict(os.environ, NO_COLOR="1", FORCE_COLOR="0", CI="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    out = sanitize(proc.stdout.decode("utf-8", "replace"), root, p["project"])
    shutil.rmtree(root, ignore_errors=True)
    rule = EXPECTED_EXIT[p["outcome"]]
    ok = (proc.returncode in rule if isinstance(rule, set) else proc.returncode in rule[p["runner"]]
          if isinstance(rule, dict) else proc.returncode != 0)
    return {"id": f"run-{p['index']:05d}", "runner": p["runner"], "outcome": p["outcome"], "theme": p["theme"],
            "project": p["project"], "n_tests": len(p["tests"]), "n_failed": len(p["failed"]),
            "n_skipped": len(p["skipped"]), "broken_kind": p["broken_kind"], "command": shown,
            "exit_code": proc.returncode, "consistent": bool(ok), "output": out}


def main():
    global PY, JS, WORK, PY_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--js-tools", type=Path, required=True, help="folder whose node_modules has jest, mocha, vitest")
    parser.add_argument("--python", type=Path, default=PY, help="a Python with pytest installed")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--per-runner", type=int, default=450)
    parser.add_argument("--ood-per-runner", type=int, default=150)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    PY, JS, WORK = args.python.resolve(), args.js_tools.resolve(), args.work.resolve()
    PY_PREFIX = Path(subprocess.run([str(PY), "-c", "import sys; print(sys.prefix)"], capture_output=True,
                                    text=True, check=True).stdout.strip()).resolve()
    plans, index = [], 0
    for runner in TRAIN_RUNNERS + OOD_RUNNERS:
        for _ in range(args.per_runner if runner in TRAIN_RUNNERS else args.ood_per_runner):
            seed = int(hashlib.sha256(f"{args.seed}:{index}".encode()).hexdigest()[:12], 16)
            plans.append(plan(index, runner, random.Random(seed)))
            index += 1
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(run_one, plans) if r]
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("runs", len(rows), "consistent", sum(r["consistent"] for r in rows))
    print(Counter((r["runner"], r["outcome"], r["consistent"]) for r in rows))
    # Anything that still names this machine needs another rule in sanitize().
    print("outputs still naming this machine:", [r["id"] for r in rows if Path.home().name in r["output"]])


if __name__ == "__main__":
    main()
