"""Version-neutral entry point for the shipped coding-state helpers.

`agent_completion_v8_run02`, the package the candidate adapter and the MBPP
preparation step imported, has never existed in this repository: the name
survived a rename in the author's working tree, while the implementation that
shipped is the one in the coding-completion preview package. The two importers
therefore name the concept instead of a run.

Deliberately a re-export rather than a move. `agent_completion_v9/train.py`
imports `canonical` by bare name after inserting its own directories on
sys.path, so relocating the implementation would break that entry point.
"""
from typed_decisions.agent_completion_v9.canonical import canonicalize
from typed_decisions.agent_completion_v9.audit_data import signature

__all__ = ['canonicalize', 'signature']
