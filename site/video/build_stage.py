"""Write site/video/stage_data.js: every number and line the demo video shows, taken from the recorded files.

Sources are the same as the website's: site/data/site_data.json (build_data.py) and site/data/capture.json
(real server, curl and Claude Code hook runs on one RTX 5060, both models).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
data = json.loads((SITE / "data" / "site_data.json").read_text(encoding="utf-8"))
cap = json.loads((SITE / "data" / "capture.json").read_text(encoding="utf-8"))
G, V1 = cap["gates"], cap["v1"]


def answers(resp):
    return {a["id"]: a for a in resp["results"][0]["answers"]}


fetch = [l for l in (SITE / "data" / "fetch_gates.log").read_text(encoding="utf-8").splitlines() if l.endswith("... ok")]
ready = G["ready"]
keep = ("event", "port", "warmup_ms", "power_throttling_disabled", "checkpoint", "encoder", "output_token_decoding")
http = {m: answers(rec["requests"]["readme_http"]["response"]) for m, rec in (("gates", G), ("v1", V1))}
http_usage = G["requests"]["readme_http"]["response"]["usage"]
gates, v1 = data["gates"]["gates"], data["gates"]["v1"]
sp = data["speed"]
V = {
    "env": data["env"],
    "why_tokens": data["why_tokens"],
    "gate": data["hero"][0],  # pytest failure, both models
    "fetch": fetch,
    "fetch_summary": '{"variant": "coding-gates", "revision": "359aa94b1af47ba4a2b2f969e99f40516cc3054e", "tensors": 343, "dtype": "BF16", …}',
    "ready": json.dumps({k: ready[k] for k in keep})[:-1] + ", …}",
    "load_seconds": G["load_seconds"],
    "http": {
        "request": G["requests"]["readme_http"]["request"],
        "gates": {"done": http["gates"]["done"]["probability"], "next": http["gates"]["next"]["distribution"]},
        "v1": {"done": http["v1"]["done"]["probability"], "next": http["v1"]["next"]["distribution"]},
        "usage": http_usage, "median_ms": G["requests"]["readme_http"]["wall_ms_median"],
    },
    "hook": data["hook"],
    "trie": data["trie"],
    "speed": {"p50": [sp["before"]["p50"], sp["after"]["p50"]], "tp": [sp["before"]["throughput"], sp["after"]["throughput"]],
              "tokens": [sp["before"]["tokens"], sp["after"]["tokens"]]},
    "grid": [[c["a"], c["b"]] for c in data["grid"]["test"]],
    "results": {
        "fp": [v1["test"]["false_pass"], gates["test"]["false_pass"], gates["test"]["negatives"]],
        "ood_fp": [v1["ood"]["false_pass"], gates["ood"]["false_pass"], gates["ood"]["negatives"]],
        "ood_fb": [v1["ood"]["false_block"], gates["ood"]["false_block"], gates["ood"]["n"] - gates["ood"]["negatives"]],
        "para": [sum(v1["test"]["final_paraphrase"].values()) / 5, sum(gates["test"]["final_paraphrase"].values()) / 5],
        "typed": [v1["typed"]["accuracy"], gates["typed"]["accuracy"]],
    },
}
assert V["results"]["fp"] == [123, 0, 348] and round(V["results"]["para"][1], 3) == 0.956
(HERE / "stage_data.js").write_text("const V = " + json.dumps(V, ensure_ascii=False) + ";\n", encoding="utf-8", newline="\n")
print("wrote stage_data.js", (HERE / "stage_data.js").stat().st_size, "bytes")
print("http", V["http"]["gates"], V["http"]["v1"], V["http"]["usage"]["wall_ms"], V["http"]["median_ms"])
