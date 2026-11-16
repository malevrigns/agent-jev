"""Assemble the AgentJev website (docs/index.html + docs/guide.html + docs/media/fonts) from recorded data.

Every number on the page comes from site/data/site_data.json (built by build_data.py from real runs),
the serving benchmark, and the video's own timeline. Run build_data.py first, and the video render
(site/video/render.js, which writes docs/media and site/video/timeline.json) before the final build so the
chapters and file size are real. Needs `npm ci` in site/ for the fonts, and git history for the old guide.
"""
import html
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "docs"
FONTS = HERE / "node_modules" / "@fontsource"
# docs/guide.html is the single-page manual that docs/index.html was at this commit, before the website replaced it.
GUIDE_BASE = "90b37d7ef66c1b94c1abb9a4b018b69188cb14a6"
if not FONTS.exists():
    raise SystemExit(f"missing {FONTS}; run: npm ci --prefix site")

data = json.loads((HERE / "data" / "site_data.json").read_text(encoding="utf-8"))
cap = json.loads((HERE / "data" / "capture.json").read_text(encoding="utf-8"))
bench_env = json.loads((REPO / "jev_service" / "serving_benchmark.json").read_text(encoding="utf-8"))["after"]["environment"]
G, V1 = cap["gates"], cap["v1"]
f1 = lambda x: f"{x:,.1f}"
f0 = lambda x: f"{x:,.0f}"

# ---------- the video: timeline from the render, size from the file ----------
timeline_path = HERE / "video" / "timeline.json"
if timeline_path.exists():
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
else:  # first build, before the video exists
    timeline = {"duration": 80.0, "chapters": [{"t": 0, "title": "演示视频渲染中"}]}
mp4 = OUT / "media" / "agentjev-demo.mp4"
duration = timeline["duration"]
video = {"chapters": timeline["chapters"], "duration": duration}

# ---------- data the page script reads ----------
hero = data["hero"]
for h in hero:
    if h["key"] == "hook":
        req = next(x for x in G["hook"] if x["command"] == "rm -rf ~/.ssh")["request"]
        h["display"] = "Claude Code → PreToolUse · Bash\n" + json.dumps(json.loads(req["state"]), ensure_ascii=False, indent=2)
    elif h["key"] == "client":
        h["display"] = h["gates"]["state"]
    else:
        h["display"] = h["gates"]["state"]

trace_req = G["requests"]["trace"]["request"]
trace_compact = json.dumps(trace_req["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) \
    if isinstance(trace_req["state"], (dict, list)) else trace_req["state"]

README_HTTP = G["requests"]["readme_http"]["request"]
http_body = json.dumps(README_HTTP, ensure_ascii=False)
http_pretty = json.dumps(README_HTTP, ensure_ascii=False, indent=2)
serve = lambda d: (f"python -m jev_service.server --checkpoint models/{d}/model.safetensors "
                   f"--model-path models/{d} --temperatures models/{d}/temperatures.json --port 8149")
py_client = """from agentjev_client import AgentJev

jev = AgentJev("http://127.0.0.1:8149")
result = jev.evaluate("23 tests passed, 1 failed.", [
    {"id": "done", "type": "boolean", "question": "Are all tests passing?",
     "criteria": {"true": "The suite is green.", "false": "At least one test failed."}},
    {"id": "next", "type": "choice", "question": "What is the useful next action?",
     "options": {"debug_failure": "Read the failing assertion.", "submit_patch": "Open a pull request now."}},
])
for answer in result["answers"]:
    print(answer["id"], answer["distribution"])"""
quick = {
    "install": {
        "bash": "git clone https://github.com/malevrigns/agent-jev.git\ncd agent-jev\npython -m venv .venv\n"
                "source .venv/bin/activate\npip install -r requirements.txt\npip install huggingface_hub safetensors",
        "ps": "git clone https://github.com/malevrigns/agent-jev.git\ncd agent-jev\npython -m venv .venv\n"
              ".venv\\Scripts\\Activate.ps1\npip install -r requirements.txt\npip install huggingface_hub safetensors"},
    "fetch": {"v1": "python fetch_v1_weights.py", "gates": "python fetch_v1_weights.py --variant coding-gates"},
    "serve": {"v1": serve("agentjev-v1"), "gates": serve("agentjev-coding-gates")},
    "ask": {
        "curl": "curl -s http://127.0.0.1:8149/api/evaluate \\\n  -H 'Content-Type: application/json' \\\n  -d '"
                + http_body.replace("'", "'\\''") + "'",
        "ps": "$body = @'\n" + http_pretty + "\n'@\nInvoke-RestMethod -Uri http://127.0.0.1:8149/api/evaluate -Method Post `\n"
              "  -ContentType 'application/json; charset=utf-8' `\n  -Body ([Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 8",
        "py": py_client},
    "note": {
        "v1": "默认权重，固定在 HF 修订 <code>7d43399</code>。Typed Decisions 评测、通用的门控和路由都用它。",
        "gates": "在 v1 上多训了读测试输出：测试集失败被判通过 123 → 0 / 348。代价是遇到 node 的 <code>dot</code> 报告器时，会多拦下实际通过的运行。"},
}


def abridge(raw):
    """The response body as curl prints it, keeping the decision fields and marking what is left out."""
    r = json.loads(raw)
    done, nxt = r["results"][0]["answers"]
    u = r["usage"]
    return [f'{{"api_version": "{r["api_version"]}", "model": "{r["model"]}", "results": [{{"id": "0", "answers": [',
            f'  {{"id": "done", "type": "boolean", "probability": {done["probability"]:.4f}, "value": {json.dumps(done["value"])}, …}},',
            f'  {{"id": "next", "type": "choice", "value": "{nxt["value"]}", "top_probability": {nxt["top_probability"]:.4f}, '
            f'"margin": {nxt["margin"]:.4f}, …}}]}}],',
            f' "usage": {{"input_path_tokens": {u["input_path_tokens"]}, "backbone_input_tokens": {u["backbone_input_tokens"]}, '
            f'"generated_tokens": {u["generated_tokens"]}, "wall_ms": {u["wall_ms"]}, …}}}}']


def terminal(model, rec, fetch_lines, request_out):
    ready = rec["ready"]
    keep = ("event", "port", "warmup_ms", "power_throttling_disabled", "checkpoint", "encoder", "output_token_decoding")
    ready_line = json.dumps({k: ready[k] for k in keep})[:-1] + ", …}"  # the printed line, abridged
    d = "agentjev-coding-gates" if model == "gates" else "agentjev-v1"
    steps = [{"cmd": quick["fetch"][model]}, {"out": fetch_lines, "gap": 140, "pause": 500},
             {"cmd": serve(d)}, {"out": [f"# 加载权重并预热（本机实测 {rec['load_seconds']:.1f} s，含一次 {ready['warmup_ms']:.0f} ms 的预热请求）"], "cls": "dm", "pause": 700},
             {"out": [ready_line], "cls": "ok", "pause": 700},
             {"cmd": "curl -s http://127.0.0.1:8149/api/evaluate -H 'Content-Type: application/json' -d @request.json"},
             {"out": abridge(request_out), "gap": 90, "pause": 300}]
    u = json.loads(request_out)["usage"]
    p = rec["requests"]["readme_http"]["response"]["results"][0]["answers"][0]["probability"]
    verdict = "判对了：有 1 个失败" if p < .5 else "判错了：明明有 1 个失败"
    steps.append({"out": [f"# 服务端 {u['wall_ms']:.1f} ms · 骨干 {u['input_path_tokens']} → {u['backbone_input_tokens']} token · 生成 0 token",
                          f"# P(全部通过) = {p:.3f}，{verdict}"], "cls": "hl" if p < .5 else "wr", "gap": 200})
    return steps


gates_fetch = [l for l in (HERE / "data" / "fetch_gates.log").read_text(encoding="utf-8").splitlines() if l.endswith("... ok")]
gates_fetch.append('{"variant": "coding-gates", "revision": "359aa94b1af47ba4a2b2f969e99f40516cc3054e", "tensors": 343, "dtype": "BF16", …}')
v1_fetch = data["terminal"]["v1_fetch"] + ['{"revision": "7d433994fbde17a3f0993c2f2b02fe8ca1370db1", "tensors": 343, "dtype": "BF16", …}']
assert G["curl"]["exit_code"] == 0 and json.loads(G["curl"]["stdout"])["results"] == G["requests"]["readme_http"]["response"]["results"]
v1_http = json.dumps(V1["requests"]["readme_http"]["response"], ensure_ascii=False)  # what the server writes
term = {"gates": terminal("gates", G, gates_fetch, G["curl"]["stdout"]), "v1": terminal("v1", V1, v1_fetch, v1_http)}

D = {
    "hero": hero, "video": video, "why_tokens": data["why_tokens"], "trace_state_compact": trace_compact,
    "examples": {"trace": data["examples"]["trace"]}, "trie": data["trie"], "grid": data["grid"],
    "board": data["board"], "hook": data["hook"], "quick": quick, "terminal": term,
    "builder": G["requests"]["readme_client"]["request"], "probes": data["probes"],
    "race": {"before": data["speed"]["before"]["samples"][:20], "after": data["speed"]["after"]["samples"][:20]},
}

# ---------- template values ----------
gv1, gg = data["gates"]["v1"], data["gates"]["gates"]
sp = data["speed"]
fin = lambda rep: statistics.mean(rep["test"]["final_paraphrase"].values())
grid = data["grid"]
ood_pos = sum(c["pass"] for c in grid["ood"])
ci = data["gates"]["typed_ci"]["ci95"]
mins, secs = divmod(round(duration), 60)
GITHUB = ('<svg viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
          '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
          '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 '
          '0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 '
          '3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')
MARK = ('<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><rect x="1" y="1" width="30" height="30" rx="8" fill="#141416" '
        'stroke="rgba(236,224,200,.22)"/><rect x="8" y="17" width="4" height="8" rx="1.5" fill="#a7813f"/>'
        '<rect x="14" y="8" width="4" height="17" rx="1.5" fill="#f4dba3"/><rect x="20" y="13" width="4" height="12" rx="1.5" fill="#d9b46c"/></svg>')
HOOK_JSON = """{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash|Write|Edit",
       "hooks": [{"type": "command", "command": "python /path/to/agent-jev/agentjev_hook.py"}]}
    ]
  }
}"""
values = {
    "mark": MARK, "github_icon": GITHUB, "hook_json": html.escape(HOOK_JSON, quote=False),
    "v1_fp": gv1["test"]["false_pass"], "gates_fp": gg["test"]["false_pass"], "gates_neg": gg["test"]["negatives"],
    "video_seconds": round(duration), "video_len": f"{mins}:{secs:02d}",
    "video_mb": f"{mp4.stat().st_size / 1e6:.1f}" if mp4.exists() else "—",
    "p50_after": f1(sp["after"]["p50"]), "p50_before": f1(sp["before"]["p50"]), "p50_x": f1(sp["before"]["p50"] / sp["after"]["p50"]),
    "typed_top1": f"{data['board'][0]['top1']:.2f}",
    "trace_q": len(data["examples"]["trace"]["questions"]), "trace_paths": data["examples"]["trace"]["usage"]["candidate_paths"],
    "trace_ms": f1(data["examples"]["trace"]["wall_ms_median"]),
    "trie_paths_tokens": f0(data["trie"]["path_tokens"]), "trie_tokens": f0(data["trie"]["trie_tokens"]),
    "tok_before": f0(sp["before"]["tokens"]), "tok_after": f0(sp["after"]["tokens"]), "tok_after_raw": sp["after"]["tokens"],
    "tp_before": f"{sp['before']['throughput']:.2f}", "tp_after": f1(sp["after"]["throughput"]),
    "tp_x": f1(sp["after"]["throughput"] / sp["before"]["throughput"]),
    "res_before": f0(sp["resident"]["before"]), "res_after": f0(sp["resident"]["after"]),
    "gpu_env": f"PyTorch {bench_env['torch']} · Transformers {bench_env['transformers']}",
    "runs_neg": sum(c["k"] == "r" for c in grid["test"]), "sums_neg": sum(c["k"] == "s" for c in grid["test"]),
    "ood_n": len(grid["ood"]), "ood_neg": gg["ood"]["negatives"], "ood_pos": ood_pos,
    "ood_fp_v1": gv1["ood"]["false_pass"], "ood_fp_g": gg["ood"]["false_pass"],
    "ood_fb_v1": gv1["ood"]["false_block"], "ood_fb_g": gg["ood"]["false_block"],
    "readme_state": html.escape(data["gates"]["readme_example"]["state"]),
    "readme_v1": f"{data['gates']['readme_example']['v1']:.2f}", "readme_g": f"{data['gates']['readme_example']['gates']:.2f}",
    "para_v1": f"{fin(gv1) * 100:.1f}%", "para_g": f"{fin(gg) * 100:.1f}%",
    "typed_v1": f"{gv1['typed']['accuracy'] * 100:.2f}%", "typed_g": f"{gg['typed']['accuracy'] * 100:.2f}%",
    "typed_diff": f"+{data['gates']['typed_ci']['difference'] * 100:.2f}", "typed_ci": f"[+{ci[0] * 100:.2f}, +{ci[1] * 100:.2f}]",
    "terse_g": f"{gg['test']['final_paraphrase']['terse'] * 100:.1f}%",
    "board_ci": data["board_ci"], "captured": data["env"]["captured"],
}
assert values["v1_fp"] == 123 and values["gates_fp"] == 0 and values["para_v1"] == "78.4%" and values["para_g"] == "95.6%"
assert ood_pos == gg["ood"]["n"] - gg["ood"]["negatives"]

body = (HERE / "src" / "body.html").read_text(encoding="utf-8")
missing = set(re.findall(r"{{(\w+)}}", body)) - set(values)
assert not missing, missing
body = re.sub(r"{{(\w+)}}", lambda m: str(values[m.group(1)]), body)

# ---------- fonts: self-hosted latin subsets, CJK falls through to the system ----------
font_faces, font_files = [], []
for family, pkg, weights in (("Inter", "inter", (400, 500, 600, 700, 800)), ("JetBrains Mono", "jetbrains-mono", (400, 500, 600))):
    css = (FONTS / pkg / "400.css").read_text(encoding="utf-8")  # every subset; take the latin block's range
    block = re.search(r"/\* " + pkg + r"-latin-400-normal \*/\s*@font-face\s*{([^}]*)}", css).group(1)
    urange = re.search(r"unicode-range:\s*([^;]+);", block).group(1).strip()
    for w in weights:
        name = f"{pkg}-latin-{w}-normal.woff2"
        font_files.append((FONTS / pkg / "files" / name, name))
        font_faces.append(f"@font-face{{font-family:'{family}';font-style:normal;font-display:swap;font-weight:{w};"
                          f"src:url(media/fonts/{name}) format('woff2');unicode-range:{urange}}}")

css = "\n".join(font_faces) + "\n" + (HERE / "src" / "style.css").read_text(encoding="utf-8")
js = "\n".join((HERE / "src" / f"app{i}.js").read_text(encoding="utf-8") for i in range(1, 5))
desc = ("AgentJev-0.6B：给 AI Agent 用的决策模型。把状态和写好的问题交给它，一次前向返回每个选项的概率，"
        "生成 token 数为 0。测试门控、工具路由、风险评分。")
favicon = ("data:image/svg+xml," + MARK.replace('class="mark" ', 'xmlns="http://www.w3.org/2000/svg" ')
           .replace("#", "%23").replace('"', "'").replace("<", "%3C").replace(">", "%3E"))
head = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>AgentJev · Agent 的反射弧</title>
<meta name="description" content="{html.escape(desc)}">
<meta name="theme-color" content="#09090a">
<meta property="og:type" content="website">
<meta property="og:title" content="AgentJev · Agent 的反射弧">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:url" content="https://malevrigns.github.io/agent-jev/">
<meta property="og:image" content="https://malevrigns.github.io/agent-jev/media/agentjev-demo-poster.jpg">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{favicon}">
<link rel="preload" href="media/fonts/inter-latin-400-normal.woff2" as="font" type="font/woff2" crossorigin>
<script>document.documentElement.classList.add("js")</script>
<style>
{css}
</style>
</head>
<body>
"""
page = (head + body + "\n<script>\nconst D = " + json.dumps(D, ensure_ascii=False, separators=(",", ":"))
        + ";\n" + js + "\n</script>\n</body>\n</html>\n")


def write(path, text):
    """Write with LF line ends, as committed; a file that differs only by a Windows checkout's CRLF is left alone."""
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8", newline="\n")


(OUT / "media" / "fonts").mkdir(parents=True, exist_ok=True)
write(OUT / "index.html", page)
for src, name in font_files:
    shutil.copyfile(src, OUT / "media" / "fonts" / name)
write(OUT / "media" / "fonts" / "Inter-OFL.txt", (FONTS / "inter" / "LICENSE").read_text(encoding="utf-8"))
write(OUT / "media" / "fonts" / "JetBrainsMono-OFL.txt", (FONTS / "jetbrains-mono" / "LICENSE").read_text(encoding="utf-8"))

# ---------- the previous single-page guide, kept as the reference manual ----------
shown = subprocess.run(["git", "-C", str(REPO), "show", f"{GUIDE_BASE}:docs/index.html"], capture_output=True)
if shown.returncode:  # a shallow clone does not have it: git fetch --unshallow
    raise SystemExit(f"git show {GUIDE_BASE[:7]}:docs/index.html failed: {shown.stderr.decode(errors='replace').strip()}")
old = shown.stdout.decode("utf-8")
stale = "# 注意：客户端会把 dict 状态缩进序列化后再发送，见下文“Python 客户端”。"
assert old.count(stale) == 1
guide = old.replace(stale, "# dict 状态原样发送，由服务端按键排序、紧凑序列化，见下文“Python 客户端”。")
guide = guide.replace("<title>AgentJev 使用手册</title>", "<title>AgentJev 参考手册</title>", 1)
blocking = re.search(r'<link rel="stylesheet" href="(https://fonts\.googleapis\.com/[^"]+)">', guide)
guide = guide.replace(blocking.group(0), f'<link rel="stylesheet" href="{blocking.group(1)}" media="print" onload="this.media=\'all\'">', 1)
meta = '<p class="meta">权重：v1 固定修订'
assert guide.count(meta) == 1
guide = guide.replace(meta, '<p class="meta"><a href="index.html">← 返回 AgentJev 官网</a><br><br>权重：v1 固定修订', 1)
write(OUT / "guide.html", guide)
print(f"wrote {OUT / 'index.html'} ({len(page.encode()) / 1024:.0f} KiB), guide.html, {len(font_files)} fonts;"
      f" video {'present' if mp4.exists() else 'missing'}, {len(video['chapters'])} chapters, {duration:.1f} s")
