"""Guard the tracked tree against imports that no tracked file provides.

Issue #3 was a tracked module importing a package that never existed in any of
the sixteen commits, so `python -m unittest discover` died during collection.
This check is static and torch-free on purpose: it has to run on a checkout that
has not installed the GPU stack.
"""
import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_PACKAGES = ('agentjev', 'jev_service', 'typed_decisions')
SKIPPED_DIRECTORIES = {'__pycache__', '.git', '.venv', 'venv', 'env', 'data',
                       'models', 'outputs', 'prepared'}


def imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def is_provided(dotted):
    target = REPO_ROOT.joinpath(*dotted.split('.'))
    return target.is_dir() or target.with_suffix('.py').is_file()


class ImportGraphTests(unittest.TestCase):
    def test_first_party_imports_exist_in_the_repository(self):
        unresolved = []
        for path in sorted(REPO_ROOT.rglob('*.py')):
            if SKIPPED_DIRECTORIES & set(path.parts) or path == Path(__file__):
                continue
            for line, dotted in imported_modules(ast.parse(path.read_text(encoding='utf-8'))):
                if dotted.split('.')[0] in TRACKED_PACKAGES and not is_provided(dotted):
                    unresolved.append(f'{path.relative_to(REPO_ROOT).as_posix()}:{line} -> {dotted}')
        self.assertEqual(unresolved, [], 'imports that no tracked file provides:\n' + '\n'.join(unresolved))
