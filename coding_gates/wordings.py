"""Ways to ask the coding-gate questions and to wrap the state, used only for training.

Trained on one fixed wording, the model learns that wording more than the task. vary() rewrites a
training request with a wording drawn from these pools: English and Chinese, with and without
criteria, and inverted gates that ask whether something is wrong (their answer is flipped).
No evaluation wording (evaluate.PARAPHRASES, evaluate.FINAL_PARAPHRASES, evaluate.PROBE_Q) is
here; tests/test_coding_gates.py checks that.
"""
import copy

# (question, criteria or None, inverted)
GATE = [
    ("Have all tests passed?", {"true": "At least one test ran and every test that ran passed.",
                                "false": "A test failed or raised an error, the tests could not load, or no tests ran."}, False),
    ("Did the test run succeed?", {"true": "Tests ran and all of them passed.",
                                   "false": "Something failed or errored, the tests did not load, or none ran."}, False),
    ("Did the tests come back clean?", None, False),
    ("Did the suite pass?", {"true": "Yes: at least one test ran and nothing failed.",
                             "false": "No: a failure, an error, a load error, or zero tests."}, False),
    ("Can the agent commit this change based on the test results?",
     {"true": "Commit: the tests ran and all passed.",
      "false": "Do not commit: tests failed, errored, could not load, or did not run."}, False),
    ("Is the test result a clean pass?", {"true": "A clean pass.", "false": "Not a clean pass."}, False),
    ("Did the run execute tests, and did they all succeed?", None, False),
    ("Is the change verified by a passing test run?",
     {"true": "Verified: tests ran and passed.", "false": "Not verified: failing, erroring, unloadable, or missing tests."}, False),
    ("测试全部通过了吗？", {"true": "运行了测试，并且全部通过。", "false": "有失败、报错、加载错误，或没有运行测试。"}, False),
    ("这次测试运行成功了吗？", None, False),
    ("根据测试结果，可以提交这次修改吗？", {"true": "可以，测试运行了且全部通过。", "false": "不可以，有失败、报错、加载错误或没有运行测试。"}, False),
    ("Does this test run show a problem?", {"true": "Yes: a failure, an error, a load error, or no tests ran.",
                                            "false": "No: tests ran and all passed."}, True),
    ("Did any test fail or error, or did no tests run at all?", None, True),
    ("Does the agent still need to fix something before the tests pass?",
     {"true": "Yes, something is still failing or broken.", "false": "No, every test passed."}, True),
    ("Is the test run broken or failing?", {"true": "Failing, erroring, unloadable, or empty.", "false": "Passing."}, True),
    ("测试结果里有问题吗？", {"true": "有：失败、报错、无法加载，或没有运行任何测试。", "false": "没有：测试运行了并且全部通过。"}, True),
    ("这次测试需要修复吗？", None, True),
]

# (question, descriptions for commit, fix_assertion, fix_exception, fix_load, find_tests)
NEXT = [
    ("What should the coding agent do next?", None),  # the build_dataset wording
    ("Pick the agent's next step.", ["Everything passed; commit.",
                                     "An assertion failed; compare expected and actual and fix the code.",
                                     "The code raised an exception; fix that error.",
                                     "A syntax or import error stopped the tests from loading; fix it first.",
                                     "No tests ran; check test discovery or write tests."]),
    ("Given this output, what is the most useful next action?",
     ["Ship it: the change is done.", "Fix the logic that makes an expected value differ.",
      "Fix the crash: the code under test throws.", "Repair the module so the tests can even be collected.",
      "Make the runner find tests."]),
    ("下一步应该做什么？", ["测试全部通过，提交修改。", "有断言失败，对比期望值和实际值后修复代码。", "被测代码抛出异常，修复报出的错误。",
                     "语法或导入错误导致测试无法加载，先修复它。", "没有找到测试，检查测试文件命名或补充测试。"]),
]

COUNT = [
    ("How many tests failed or raised an error?", None),  # the build_dataset wording
    ("Count the tests that failed or errored.", ["None.", "One.", "Two to five.", "Six or more."]),
    ("有多少个测试失败或报错？", ["没有。", "正好一个。", "两到五个。", "六个或更多。"]),
]

TASKS = ["Implement {fn} in the {mod} module.", "Refactor {mod} for readability.", "Add input validation to {fn}.",
         "Update {fn} to handle edge cases.", "修复 {mod} 模块里的问题。", "为 {fn} 补充边界情况处理。"]
FUNCS = ["add", "clamp", "slugify", "discount", "word_count", "pad_left", "fib", "reverse"]
MODS = ["mathutil", "textutil", "pricing", "strings"]


def _task(rng):
    return rng.choice(TASKS).format(fn=rng.choice(FUNCS), mod=rng.choice(MODS))


def _state(state, rng):
    """Same facts under other key names, sometimes with the task the agent was given."""
    if isinstance(state, str):
        shape = rng.random()
        if shape < .6:
            return state
        return {"test_output": state} if shape < .8 else {"task": _task(rng), "test_output": state}
    command, output, code = state["command"], state["output"], state.get("exit_code")
    shape = rng.randrange(4)
    if shape == 0:
        out = dict(state)
    elif shape == 1:
        out = {"cmd": command, "test_output": output}
        if code is not None:
            out["exit_status"] = code
    elif shape == 2:
        out = {"ran": command, "log": output}
        if code is not None:
            out["returncode"] = code
    else:  # shaped like a coding agent's tool call and result
        out = {"tool": "Bash", "input": {"command": command}, "result": {"stdout": output}}
        if code is not None:
            out["result"]["exit_code"] = code
    if rng.random() < .3:
        out["task"] = _task(rng)
    return out


def vary(request, targets, rng, keep_original=.25):
    """A reworded copy of one gate request and its targets (inverted gates flip their target)."""
    request, targets = copy.deepcopy(request), list(targets)
    request["state"] = _state(request["state"], rng)
    for i, q in enumerate(request["questions"]):
        if rng.random() < keep_original:
            continue
        if q["id"] == "passed":
            text, criteria, inverted = rng.choice(GATE[1:])
            q["question"] = text
            q.pop("criteria", None)
            if criteria:
                q["criteria"] = dict(criteria)
            if inverted:
                targets[i] = targets[i][::-1]
        elif q["id"] == "next":
            text, descriptions = rng.choice(NEXT[1:])
            q["question"] = text
            q["options"] = dict(zip(q["options"], descriptions))
        elif q["id"] == "failures":
            text, levels = rng.choice(COUNT[1:])
            q["question"], q["levels"] = text, list(levels)
    return request, targets
