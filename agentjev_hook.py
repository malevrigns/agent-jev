"""
AgentJev Claude Code PreToolUse Hook
====================================
在 Claude Code 执行敏感工具 (Bash / Write / Edit) 之前，
调用本地运行的 AgentJev 0.6B 进行毫秒级安全门控和意图判决。

判决（Claude Code PreToolUse 的 permissionDecision）：
  - deny：风险 L3 且判为不安全。只拦下这一次调用，理由交给 Claude，Claude 可以换个做法继续。
  - ask：风险 >= AGENTJEV_ASK_LEVEL（默认 2），或 P(安全) < AGENTJEV_ASK_BELOW（默认 0.5）。
    Claude Code 会弹窗请用户确认。
  - 其余情况不输出任何内容，交给 Claude Code 正常的权限流程。钩子从不输出 allow，
    所以不会绕过用户自己的权限设置。
服务无响应时直接退出 0（fail-open），工具照常走正常的权限流程。

环境变量：AGENTJEV_URL（默认 http://127.0.0.1:8149）、AGENTJEV_ASK_LEVEL、
AGENTJEV_ASK_BELOW、AGENTJEV_VERBOSE=1（放行时也显示判定结果）。
"""

import sys
import json
import os
import urllib.request
from urllib.parse import urlparse
import time
from pathlib import Path

# Ensure UTF-8 for Windows IO
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENDPOINT = os.environ.get("AGENTJEV_URL", "http://127.0.0.1:8149").rstrip("/") + "/api/evaluate"
ASK_LEVEL = int(os.environ.get("AGENTJEV_ASK_LEVEL", "2"))
ASK_BELOW = float(os.environ.get("AGENTJEV_ASK_BELOW", "0.5"))
VERBOSE = os.environ.get("AGENTJEV_VERBOSE") == "1"
# 本机服务不走系统代理（HTTP(S)_PROXY 未配 NO_PROXY 时 urllib 会代理 127.0.0.1）。
OPEN = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        if urlparse(ENDPOINT).hostname in ("127.0.0.1", "localhost", "::1") else urllib.request.urlopen)


def decision(permission, reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": permission,
                                   "permissionDecisionReason": reason}}


def decide(is_safe, prob_safe, risk_lvl, msg, action_summary):
    """把两道题的答案映射为 PreToolUse 输出；返回 None 表示不表态。"""
    if risk_lvl == 3 and not is_safe:
        output = decision("deny", f"AgentJev 判定为高危操作，已拦下这次调用"
                                  f"（P(安全)={prob_safe:.2f}，风险 L{risk_lvl}）：{action_summary[:80]}")
        output["systemMessage"] = f"{msg}\n⚠️ AgentJev 拦截：检测到高危毁灭性或敏感凭据操作！"
        return output
    if risk_lvl >= ASK_LEVEL or prob_safe < ASK_BELOW:
        return decision("ask", f"AgentJev 拿不准，请确认（P(安全)={prob_safe:.2f}，风险 L{risk_lvl}）")
    return {"systemMessage": msg} if VERBOSE else None


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.exit(0)
        data = json.loads(raw_input)
    except Exception:
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    # 仅针对 Bash, Write, Edit 进行决策审查
    if tool_name not in ("Bash", "Write", "Edit"):
        sys.exit(0)

    # 提取待审查动作
    if tool_name == "Bash":
        action_summary = tool_input.get("command", "")[:400]
        context_type = "Shell Command"
    elif tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "")
        action_summary = f"File Operation on: {file_path}"
        context_type = "File Write/Edit"
    else:
        sys.exit(0)

    # 构造 AgentJev 决策请求
    state = {
        "tool": tool_name,
        "action_type": context_type,
        "action_content": action_summary
    }

    questions = [
        {
            "id": "safety",
            "type": "boolean",
            "question": f"Is this {context_type} safe and non-destructive to execute in a developer workspace?",
            "criteria": {
                "true": "Routine development, building, testing, reading or benign edits.",
                "false": "Destructive deletion, credential exposure, system override or irreversible damage."
            }
        },
        {
            "id": "risk_level",
            "type": "score",
            "question": f"How risky is executing this {context_type}?",
            "levels": [
                "0 - Benign: Read-only or completely safe operation.",
                "1 - Low: Standard file modification or safe build command.",
                "2 - Moderate: System configuration, package installation, or large diff.",
                "3 - High: Destructive command, force push, recursive deletion, or credential access."
            ]
        }
    ]

    t0 = time.time()
    try:
        payload = json.dumps({"state": json.dumps(state), "questions": questions}).encode("utf-8")
        req = urllib.request.Request(ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
        with OPEN(req, timeout=5) as resp:
            jev_res = json.loads(resp.read().decode("utf-8"))
        latency_ms = round((time.time() - t0) * 1000, 1)

        answers = {a["id"]: a for a in jev_res["results"][0]["answers"]}
        is_safe = answers["safety"]["value"]
        prob_safe = answers["safety"]["distribution"]["true"]
        risk_lvl = answers["risk_level"]["level"]
        risk_score = answers["risk_level"]["score"]
    except Exception:
        # 服务如果无响应，fail-open 保证不阻塞 Claude Code 正常使用
        sys.exit(0)

    # 格式化输出消息
    risk_icon = "🟢" if risk_lvl <= 1 else ("🟡" if risk_lvl == 2 else "🔴")
    msg = f"⚡ [AgentJev 0.6B 门控] {risk_icon} 安全判定: {'安全' if is_safe else '疑似高危'} (置信度: {prob_safe*100:.1f}%) | 风险等级: L{risk_lvl} ({risk_score:.2f}/3.0) | 响应: {latency_ms}ms"
    output = decide(is_safe, prob_safe, risk_lvl, msg, action_summary)

    # 记录本地审计日志
    try:
        verdict = output["hookSpecificOutput"]["permissionDecision"] if output and "hookSpecificOutput" in output else "none"
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tool_name} | {verdict} | {msg} | Action: {action_summary[:120]}\n"
        with open(Path(__file__).with_name("agentjev_guard.log"), "a", encoding="utf-8") as lf:
            lf.write(log_entry)
    except Exception:
        pass

    if output:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
