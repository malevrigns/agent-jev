"""PreToolUse decisions from agentjev_hook. Standard library only, no server."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import agentjev_hook
from agentjev_hook import decide

HOOK = Path(agentjev_hook.__file__)


def permission(output):
    return output["hookSpecificOutput"]["permissionDecision"] if output else None


class DecideTests(unittest.TestCase):
    def test_high_risk_and_unsafe_denies_only_this_call(self):
        output = decide(False, 0.35, 3, "msg", "rm -rf ~/.ssh")
        self.assertEqual(permission(output), "deny")
        self.assertEqual(output["hookSpecificOutput"]["hookEventName"], "PreToolUse")
        self.assertIn("rm -rf ~/.ssh", output["hookSpecificOutput"]["permissionDecisionReason"])
        # continue:false would stop Claude entirely instead of refusing one call.
        self.assertNotIn("continue", output)

    def test_uncertain_calls_ask_the_user(self):
        self.assertEqual(permission(decide(True, 0.64, 3, "msg", "git push --force")), "ask")
        self.assertEqual(permission(decide(True, 0.90, 2, "msg", "apt install x")), "ask")
        self.assertEqual(permission(decide(False, 0.40, 1, "msg", "chmod 777 x")), "ask")

    def test_confident_low_risk_has_no_opinion(self):
        self.assertIsNone(decide(True, 0.90, 1, "msg", "ls -la src/"))

    def test_never_allows(self):
        # "allow" would bypass the user's own permission prompts.
        for args in [(True, 0.99, 0), (True, 0.6, 1), (False, 0.1, 3), (True, 0.5, 2)]:
            self.assertNotEqual(permission(decide(*args, "msg", "cmd")), "allow")


class ProcessTests(unittest.TestCase):
    def run_hook(self, payload, url="http://127.0.0.1:9"):
        env = {"AGENTJEV_URL": url, "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), capture_output=True,
                              text=True, encoding="utf-8", env=env, timeout=30)

    def test_unreachable_server_fails_open_silently(self):
        result = self.run_hook({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        self.assertEqual((result.returncode, result.stdout), (0, ""))

    def test_other_tools_are_ignored(self):
        result = self.run_hook({"tool_name": "Read", "tool_input": {"file_path": "x"}})
        self.assertEqual((result.returncode, result.stdout), (0, ""))


if __name__ == "__main__":
    unittest.main()
