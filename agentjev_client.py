"""
AgentJev Client SDK
===================
与 TypeSafe Jev / NanoJev 规范对齐的 Python 客户端。
用于在真实 Agent 循环、CI/CD 门控、运维自动化或多轮对话中快速执行 System One 决策。
"""

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Union


class AgentJevError(RuntimeError):
    """服务端拒绝请求（4xx）或推理失败（5xx）时抛出，message 为服务端给出的原因。"""

    def __init__(self, status: int, message: str):
        super().__init__(f"AgentJev HTTP {status}: {message}")
        self.status = status
        self.message = message


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _opener(endpoint: str):
    # 本机服务不走系统代理：设置了 HTTP(S)_PROXY 却没设 NO_PROXY 时，urllib 会把 127.0.0.1 也送进代理。
    if urlparse(endpoint).hostname in _LOOPBACK:
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    return urllib.request.urlopen


def _state(state: Any) -> Any:
    # dict / list 原样发送：服务端会按键排序、紧凑序列化，与基准测试的输入一致。
    # 在客户端先缩进序列化会改变模型输入，从而改变概率。
    return state if isinstance(state, (dict, list, str)) else str(state)


class AgentJev:
    def __init__(self, endpoint: str = "http://127.0.0.1:8149", timeout: float = 30):
        self.endpoint = endpoint.rstrip("/")
        self.eval_url = f"{self.endpoint}/api/evaluate"
        self.info_url = f"{self.endpoint}/api/info"
        self.timeout = timeout
        self._open = _opener(self.endpoint)

    def info(self) -> Dict[str, Any]:
        """获取当前服务状态与模型信息"""
        req = urllib.request.Request(self.info_url)
        with self._open(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.eval_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with self._open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                message = json.loads(body).get("error", body)
            except ValueError:
                message = body
            raise AgentJevError(exc.code, message) from None

    def evaluate(self, state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        低阶 API：向 AgentJev 提交一次状态及多个决策问题。
        同一状态的所有问题放在一次调用里，服务端只编码一次状态。
        返回原始 answers 列表及耗时指标。
        """
        t0 = time.time()
        data = self._post({"state": _state(state), "questions": questions})
        wall_ms = (time.time() - t0) * 1000

        result = data["results"][0]
        result["network_wall_ms"] = round(wall_ms, 2)
        result["server_usage"] = data.get("usage", {})
        return result

    def evaluate_many(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量 API：一次提交多个状态，每项为 {"state": ..., "questions": [...]}，可带 "id"。
        单次最多 32 个状态、128 道题、1,024 条候选路径。返回与输入同序的 results。
        """
        payload = {"requests": [dict(item, state=_state(item["state"])) for item in requests]}
        return self._post(payload)["results"]

    def decide_boolean(self, state: Any, question: str, criteria: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        高阶 API 1：真假判断 (Boolean / noul)
        用于任务是否完成、测试是否通过、是否需要回滚或人工介入等场景。
        """
        q = {
            "id": "bool_decision",
            "type": "boolean",
            "question": question,
            "criteria": criteria or {"true": "The statement is true.", "false": "The statement is false."}
        }
        res = self.evaluate(state, [q])
        ans = res["answers"][0]
        return {
            "decision": ans["value"],            # True 或 False
            "prob_true": ans["distribution"]["true"],
            "prob_false": ans["distribution"]["false"],
            "confidence": max(ans["distribution"].values()),
            "wall_ms": res["network_wall_ms"]
        }

    def decide_choice(self, state: Any, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        """
        高阶 API 2：动态候选决策 (Choice)
        从多个可行动作中选出最优方案，返回命中选项及置信度差 (margin)。
        """
        q = {
            "id": "choice_decision",
            "type": "choice",
            "question": question,
            "options": options
        }
        res = self.evaluate(state, [q])
        ans = res["answers"][0]
        dist = ans["distribution"]
        sorted_options = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top_key, top_prob = sorted_options[0]
        second_prob = sorted_options[1][1] if len(sorted_options) > 1 else 0.0

        return {
            "best_action": top_key,
            "description": options.get(top_key, ""),
            "probability": top_prob,
            "margin": round(top_prob - second_prob, 4),  # 置信优势 (大于 0.2 通常表明极其坚决)
            "distribution": dist,
            "wall_ms": res["network_wall_ms"]
        }

    def score(self, state: Any, question: str, levels: List[str]) -> Dict[str, Any]:
        """
        高阶 API 3：等级评分 (Score)
        返回离散等级及连续期望得分 (0.0 ~ N-1.0)。
        """
        q = {
            "id": "score_decision",
            "type": "score",
            "question": question,
            "levels": levels
        }
        res = self.evaluate(state, [q])
        ans = res["answers"][0]
        return {
            "expected_score": round(ans["score"], 3),
            "level": ans["level"],
            "level_description": levels[ans["level"]],
            "distribution": ans["distribution"],
            "wall_ms": res["network_wall_ms"]
        }
