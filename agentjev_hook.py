"""
AgentJev Claude Code PreToolUse Hook
====================================
在 Claude Code 执行敏感工具 (Bash / Write / Edit) 之前，
调用本地运行的 AgentJev 0.6B 进行毫秒级安全门控和意图判决。
"""

import sys
import json
import urllib.request
import time
from pathlib import Path

# Ensure UTF-8 for Windows IO
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENDPOINT = "http://127.0.0.1:8149/api/evaluate"

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
        with urllib.request.urlopen(req, timeout=5) as resp:
            jev_res = json.loads(resp.read().decode("utf-8"))
        latency_ms = round((time.time() - t0) * 1000, 1)

        answers = {a["id"]: a for a in jev_res["results"][0]["answers"]}
        is_safe = answers["safety"]["value"]
        prob_safe = answers["safety"]["distribution"]["true"]
        risk_lvl = answers["risk_level"]["level"]
        risk_score = answers["risk_level"]["score"]

        # 格式化输出消息
        risk_icon = "🟢" if risk_lvl <= 1 else ("🟡" if risk_lvl == 2 else "🔴")
        msg = f"⚡ [AgentJev 0.6B 门控] {risk_icon} 安全判定: {'安全' if is_safe else '疑似高危'} (置信度: {prob_safe*100:.1f}%) | 风险等级: L{risk_lvl} ({risk_score:.2f}/3.0) | 响应: {latency_ms}ms"

        # 记录本地审计日志
        try:
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tool_name} | {msg} | Action: {action_summary[:120]}\n"
            with open(Path(__file__).with_name("agentjev_guard.log"), "a", encoding="utf-8") as lf:
                lf.write(log_entry)
        except Exception:
            pass

        # 如果 AgentJev 判定为严重高危 (Level 3 且不安全)
        if risk_lvl == 3 and not is_safe:
            output = {
                "systemMessage": f"{msg}\n⚠️ AgentJev 拦截阻断：检测到高危毁灭性或敏感凭据操作！",
                "continue": False,
                "stopReason": f"AgentJev 安全拦截: {action_summary[:80]}"
            }
        else:
            output = {
                "systemMessage": msg,
                "continue": True
            }

        print(json.dumps(output, ensure_ascii=False))

    except Exception:
        # 服务如果无响应，fail-open 保证不阻塞 Claude Code 正常使用
        sys.exit(0)

if __name__ == "__main__":
    main()
