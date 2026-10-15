"""Collect every number the website and the demo video show, from recorded files only.

Sources: site/data/capture.json (real server + hook, both models), site/data/fetch_gates.log and
site/data/fetch_v1.log (real `fetch_v1_weights.py` runs), the repo's coding_gates/results (held-out reports and
per-question predictions), jev_service/serving_benchmark.json (the 400-case serving benchmark behind the
README table), the README's Typed Decisions table, and the release file's safetensors header.

Needs the coding-gates tokenizer and model.safetensors (`python fetch_v1_weights.py --variant coding-gates`)
and coding_gates/data (`python coding_gates/build_dataset.py --model-path models/agentjev-coding-gates`);
only the safetensors header is read, so no GPU. Writes site/data/site_data.json, or with --check compares.
"""
import argparse
import gzip
import json
import math
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model-dir", type=Path, default=REPO / "models" / "agentjev-coding-gates",
                help="the coding-gates release: tokenizer files and model.safetensors (default: %(default)s)")
ap.add_argument("--check", action="store_true",
                help="rebuild in memory and exit 1 if it differs from the committed site/data/site_data.json")
args = ap.parse_args()
test_requests = REPO / "coding_gates" / "data" / "test_requests.jsonl"
for need, how in ((args.model_dir / "model.safetensors", "python fetch_v1_weights.py --variant coding-gates"),
                  (test_requests, "python coding_gates/build_dataset.py --model-path models/agentjev-coding-gates")):
    if not need.exists():
        sys.exit(f"missing {need}; run: {how}")
sys.path.insert(0, str(REPO))
from agentjev.data import build_tries  # noqa: E402
from jev_service.contract import encode_paths, prepare  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl_gz(path):
    return [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]


cap = load(HERE / "data" / "capture.json")
G, V1 = cap["gates"], cap["v1"]


def answers(request, response):
    prepared = prepare(request)[0]
    out = []
    for q, a in zip(prepared["questions"], response["results"][0]["answers"]):
        item = {"id": q["id"], "type": q["type"], "text": q["text"], "keys": q["keys"],
                "candidates": q["candidates"], "probs": [round(a["distribution"][k], 4) for k in q["keys"]]}
        for key in ("value", "level", "score", "margin", "top_probability"):
            if key in a:
                item[key] = round(a[key], 4) if isinstance(a[key], float) else a[key]
        out.append(item)
    return out


def usage(response):
    u = response["usage"]
    return {k: u[k] for k in ("questions", "candidate_paths", "input_path_tokens", "backbone_input_tokens",
                              "generated_tokens", "wall_ms")}


def state_text(state):
    if isinstance(state, dict) and set(state) >= {"command", "output"}:
        return f"$ {state['command']}\n{state['output']}\n[exit {state['exit_code']}]"
    if isinstance(state, str):
        try:
            return json.dumps(json.loads(state), ensure_ascii=False, indent=2)
        except ValueError:
            return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def scene(rec):
    return {"state": state_text(rec["request"]["state"]), "questions": answers(rec["request"], rec["response"]),
            "usage": usage(rec["response"]), "wall_ms_median": rec.get("wall_ms_median")}


hook_by = {h["command"]: h for h in G["hook"]}
v1_hook_by = {h["command"]: h for h in V1["hook"]}
deny = hook_by["rm -rf ~/.ssh"]

data = {"env": cap["environment"]}
def hook_scene(model, command):
    h = next(x for x in model["hook"] if x["command"] == command)
    sc = scene({"request": h["request"], "response": h["response"]})
    sc["state"] = command
    sc["decision"] = h["decision"]
    sc["reason"] = json.loads(h["stdout"])["hookSpecificOutput"]["permissionDecisionReason"] if h["stdout"] else None
    return sc


data["hero"] = [
    {"key": "gate", "label": "测试门控", "source": "pytest -q --tb=short", "gates": scene(G["requests"]["coding_fail"]),
     "v1": scene(V1["requests"]["coding_fail"])},
    {"key": "hook", "label": "工具门控", "source": "Claude Code · Bash", "gates": hook_scene(G, "rm -rf ~/.ssh"),
     "v1": hook_scene(V1, "rm -rf ~/.ssh")},
    {"key": "client", "label": "README 示例", "source": "agentjev_client.py", "gates": scene(G["requests"]["readme_client"]),
     "v1": scene(V1["requests"]["readme_client"])},
]
data["examples"] = {k: scene(G["requests"][k]) for k in G["requests"]}
data["examples_v1"] = {k: scene(V1["requests"][k]) for k in V1["requests"]}

data["hook"] = []
for command, h in hook_by.items():
    out = json.loads(h["stdout"]) if h["stdout"] else None
    v = v1_hook_by[command]
    data["hook"].append({"command": command, "decision": h["decision"], "p_safe": round(h["p_safe"], 3),
                         "level": h["risk_level"], "score": round(h["risk_score"], 2),
                         "reason": out["hookSpecificOutput"]["permissionDecisionReason"] if out else None,
                         "system_message": out.get("systemMessage") if out else None, "stdout": h["stdout"],
                         "server_ms": h["server_wall_ms"], "hook_ms": h["hook_process_ms"],
                         "v1": {"decision": v["decision"], "p_safe": round(v["p_safe"], 3), "level": v["risk_level"]}})

# Quickstart terminal: real lines only.
fetch_lines = [l for l in (HERE / "data" / "fetch_gates.log").read_text(encoding="utf-8").splitlines()
               if l.strip() and "Xet Storage" not in l]
ready = G["ready"]
data["terminal"] = {
    "fetch": fetch_lines,
    "ready": {k: ready[k] for k in ("event", "port", "warmup_ms", "power_throttling_disabled", "checkpoint",
                                     "checkpoint_sha256", "encoder", "max_path_tokens", "output_token_decoding")},
    "request": G["requests"]["readme_http"]["request"],
    "response": G["requests"]["readme_http"]["response"],
    "load_seconds": G["load_seconds"],
    "curl": G["curl"]["stdout"], "curl_ms": G["curl"]["elapsed_ms"], "health": G["curl_health"],
    "v1_ready": {k: V1["ready"][k] for k in ("event", "port", "warmup_ms", "power_throttling_disabled", "checkpoint",
                                               "checkpoint_sha256", "encoder", "max_path_tokens", "output_token_decoding")},
    "v1_response": V1["requests"]["readme_http"]["response"],
    "v1_fetch": [l for l in (HERE / "data" / "fetch_v1.log").read_text(encoding="utf-8").splitlines() if l.endswith("... ok")],
}

# Serving benchmark (README "Latency & throughput"): 400 test cases, one request each.
bench_repo = load(REPO / "jev_service" / "serving_benchmark.json")
before_doc, after_doc = bench_repo["before"], bench_repo["after"]
before, after = before_doc["modes"]["auto"], after_doc["modes"]["auto"]
wide_before = before["scenarios"]["wide"]
wide_after = after["scenarios"]["wide"]
data["speed"] = {
    "before": {"p50": before["case_latency"]["p50_ms"], "p90": before["case_latency"]["p90_ms"],
               "throughput": before["batched_cases_per_second"], "tokens": round(before["backbone_input_tokens_per_case"]),
               "samples": [round(x, 1) for x in before["case_latency_samples_ms"]]},
    "after": {"p50": after["case_latency"]["p50_ms"], "p90": after["case_latency"]["p90_ms"],
              "throughput": after["batched_cases_per_second"], "tokens": round(after["backbone_input_tokens_per_case"]),
              "samples": [round(x, 1) for x in after["case_latency_samples_ms"]]},
    "resident": {"before": before_doc["resident_weights_mib"], "after": after_doc["resident_weights_mib"]},
    "wide": {"before": wide_before["p50_ms"], "after": wide_after["p50_ms"], "path_tokens": wide_after["input_path_tokens"],
             "backbone_before": wide_before["backbone_input_tokens"], "backbone_after": wide_after["backbone_input_tokens"]},
    "gpu": after_doc["environment"]["gpu"],
}
assert data["speed"]["before"]["p50"] == 278.01 and data["speed"]["after"]["p50"] == 43.89

# The trace request as a token trie: one state, five questions, eighteen candidates.
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
trace_req = G["requests"]["trace"]["request"]
paths, locations, rows = encode_paths(prepare(trace_req), tok, 2048)


def lcp(seqs):
    n = 0
    while all(len(s) > n for s in seqs) and len({s[n] for s in seqs}) == 1:
        n += 1
    return n


state_len = lcp(paths)
trie = {"state": state_len, "questions": [], "path_tokens": sum(map(len, paths))}
for qi, q in enumerate(rows):
    group = [p for p, (qq, _) in zip(paths, locations) if qq == qi]
    shared = lcp(group)
    trie["questions"].append({"id": q["id"], "type": q["type"], "q": shared - state_len,
                              "cands": [len(p) - shared for p in group], "keys": q["keys"]})
nodes = build_tries(paths, [0] * len(paths), 1)[0][0]
trie["trie_tokens"] = len(nodes)
assert trie["trie_tokens"] == G["requests"]["trace"]["response"]["usage"]["backbone_input_tokens"], trie["trie_tokens"]
assert trie["path_tokens"] == G["requests"]["trace"]["response"]["usage"]["input_path_tokens"]
data["trie"] = trie
# The "writer as a switch" lane: an answer a generative model would have to decode, split by the real tokenizer.
WRITER_ANSWER = '{"all_tests_passed": false, "next_step": "fix the failing assertion in test_max_of_sample_2"}'
data["why_tokens"] = [tok.decode([t]) for t in tok.encode(WRITER_ANSWER, add_special_tokens=False)]

# Coding gates: held-out reports, read once (coding_gates/results).
R = REPO / "coding_gates" / "results"
fv1, fc, cmpr = load(R / "final_v1.json"), load(R / "final_candidate.json"), load(R / "compare.json")


def gate_summary(report, split):
    s = report["splits"][split]
    return {"gate_acc": s["by_type"]["boolean"]["accuracy"], "overall": s["overall"]["accuracy"], **s["gate"],
            "paraphrase": {k: v["accuracy"] for k, v in s["paraphrase"].items()},
            "final_paraphrase": {k: v["accuracy"] for k, v in s["final_paraphrase"].items()}}


data["gates"] = {
    model: {"test": gate_summary(rep, "test"), "ood": gate_summary(rep, "ood"),
            "typed": {"accuracy": rep["typed"]["test"]["accuracy"], "soft_ce": rep["typed"]["test"]["soft_cross_entropy"]},
            "ecosystems": sum((e["p_true"] >= .5) == e["gold_true"] for e in rep["ecosystems"]),
            "ecosystems_n": len(rep["ecosystems"])}
    for model, rep in (("v1", fv1), ("gates", fc))}
data["gates"]["typed_ci"] = {"difference": cmpr["typed_test"]["difference"], "ci95": cmpr["typed_test"]["ci95"]}
data["gates"]["readme_example"] = {
    "state": G["requests"]["readme_http"]["request"]["state"],
    "v1": V1["requests"]["readme_http"]["response"]["results"][0]["answers"][0]["probability"],
    "gates": G["requests"]["readme_http"]["response"]["results"][0]["answers"][0]["probability"]}

# Per-case grids from the per-question predictions (the gate question only).
runs = {r["id"]: r for r in jsonl_gz(REPO / "coding_gates" / "runs.jsonl.gz")}
# Paraphrase rows reuse the gate question id, so key by (set, id).
pv1 = {(p["set"], p["id"]): p for p in jsonl_gz(R / "final_v1_pred.jsonl.gz") if p["type"] == "boolean"}
pc = {(p["set"], p["id"]): p for p in jsonl_gz(R / "final_candidate_pred.jsonl.gz") if p["type"] == "boolean"}
test_states = {r["id"]: r["state"] for r in map(json.loads, open(test_requests, encoding="utf-8"))}


def reporter(cmd):
    for word in ("tap", "dot", "verbose"):
        if f"reporter={word}" in cmd or f"--reporter {word}" in cmd:
            return word
    return "default" if cmd.startswith("npx vitest") else "spec" if cmd.startswith("node") else ""


grid_test, grid_ood = [], []
for pid, a in pv1.items():
    b = pc[pid]
    case = a["case_id"]
    failing = a["target"][0] == 0
    if a["set"] == "test" and failing:
        if case.startswith("summary-"):
            grid_test.append({"k": "s", "t": test_states[case], "a": round(a["probs"][0], 3), "b": round(b["probs"][0], 3)})
        else:
            r = runs[case]
            grid_test.append({"k": "r", "t": f"{r['runner']} · {r['outcome']} · {r['command']}", "a": round(a["probs"][0], 3),
                              "b": round(b["probs"][0], 3)})
    elif a["set"] == "ood":
        r = runs[case]
        grid_ood.append({"pass": not failing, "t": f"{r['runner']} · {r['outcome']} · {r['command']}",
                         "a": round(a["probs"][0], 3), "b": round(b["probs"][0], 3)})
order = {"r": 0, "s": 1}
grid_test.sort(key=lambda c: (order[c["k"]], c["t"]))
grid_ood.sort(key=lambda c: (c["pass"], c["t"]))
assert len(grid_test) == 348 and sum(c["a"] >= .5 for c in grid_test) == 123 and sum(c["b"] >= .5 for c in grid_test) == 0
assert len(grid_ood) == 300 and sum((c["a"] >= .5) != c["pass"] for c in grid_ood if not c["pass"]) == 31
data["grid"] = {"test": grid_test, "ood": grid_ood}

# README "Typed Decisions" table.
readme = (REPO / "README.md").read_text(encoding="utf-8")
section = readme[readme.index("## Typed Decisions"):readme.index("Rows without a soft-CE number")]
board = []
for line in section.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 7 or not re.search(r"\d+(\.\d+)?%", cells[2]):
        continue
    clean = lambda s: re.sub(r"\*\*", "", s)
    board.append({"model": clean(cells[0]), "kind": cells[1], "top1": float(re.search(r"(\d+(?:\.\d+)?)%", cells[2]).group(1)),
                  "soft_ce": None if cells[3] in ("—", "-") else float(clean(cells[3]))})
assert board[0]["top1"] == 79.25 and len(board) == 8, board
data["board"] = board
data["board_ci"] = re.search(r"interval of \*\*\[([^\]]+)\]\*\*", readme).group(1)

# Parameter count from the release file's header.
with open(args.model_dir / "model.safetensors", "rb") as fh:
    header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
data["params"] = sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
data["health"] = {k: G["health"][k] for k in ("max_path_tokens", "max_choice_candidates", "types", "encoder", "temperatures")}

data["probes"] = {"v1": fv1["probes"], "gates": fc["probes"]}
data["ecosystems"] = {"v1": fv1["ecosystems"], "gates": fc["ecosystems"]}
out = HERE / "data" / "site_data.json"
text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
if args.check:
    same = out.exists() and out.read_text(encoding="utf-8") == text
    print(out, "matches the recorded inputs" if same else "differs from a rebuild; run build_data.py and commit it")
    sys.exit(0 if same else 1)
out.write_text(text, encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes | params", f"{data['params']:,}", "| trie", trie["path_tokens"], "->",
      trie["trie_tokens"], "| board", [(b["model"][:18], b["top1"]) for b in board])
