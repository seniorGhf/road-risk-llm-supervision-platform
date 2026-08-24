from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


ACTION_CODES = {
    "发布上游可变情报板预警": "VMS_WARNING",
    "调取视频或派出巡查核验": "VERIFY",
    "准备重型救援与起吊力量": "HEAVY_RESCUE",
    "临时隔离相邻车道": "LANE_ISOLATION",
    "布设低速队尾预警": "LOW_SPEED_WARNING",
    "加强上游排队传播管控": "QUEUE_PROPAGATION",
    "实施车道控制与分流": "LANE_CONTROL",
    "准备消防与危险品隔离": "FIRE_RESPONSE",
    "实施天气自适应限速": "WEATHER_SPEED_CONTROL",
}


class PolicyEngine:
    """Generate policy-grounded, human-gated operational review packets."""

    def __init__(self, policy_library: dict[str, Any]):
        self.policy_library = policy_library
        self.use_ollama = os.getenv("RISK_PLATFORM_USE_OLLAMA", "0") == "1"
        self.ollama_url = os.getenv("RISK_PLATFORM_OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("RISK_PLATFORM_OLLAMA_MODEL", "qwen3:8b")

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        deterministic = self._grounded_review(packet)
        if not self.use_ollama:
            return deterministic
        try:
            candidate = self._call_ollama(packet, deterministic)
            return self._validate_llm(candidate, deterministic)
        except Exception as exc:
            deterministic["engine"] = "policy-safety-fallback"
            deterministic["engine_note"] = f"本地大模型不可用，已切换安全兜底：{exc}"
            return deterministic

    def _grounded_review(self, packet: dict[str, Any]) -> dict[str, Any]:
        score = float(packet.get("score", 0.0))
        speed = float(packet.get("median_speed_km_h", 80.0))
        upstream = float(packet.get("upstream_traffic", 0.0))
        downstream = float(packet.get("downstream_traffic", 0.0))
        weather = str(packet.get("weather", "未知"))
        type_name = str(packet.get("type_hypothesis", "其他"))

        actions = ["发布上游可变情报板预警", "调取视频或派出巡查核验"]
        if type_name == "翻车":
            actions.extend(["准备重型救援与起吊力量", "临时隔离相邻车道"])
        elif type_name == "失火":
            actions.append("准备消防与危险品隔离")
        elif type_name in {"碰撞", "追尾", "刮擦"}:
            actions.append("实施车道控制与分流")
        if speed < 40:
            actions.append("布设低速队尾预警")
        if abs(upstream - downstream) >= 25:
            actions.append("加强上游排队传播管控")
        if weather not in {"Clear", "晴", "未知", ""}:
            actions.append("实施天气自适应限速")

        actions = list(dict.fromkeys(actions))
        if score < 0.6:
            actions = ["调取视频或派出巡查核验"]

        reason_parts = [f"监督分数 {score:.3f}"]
        if speed < 40:
            reason_parts.append(f"中位速度 {speed:.1f} km/h，存在低速冲击波迹象")
        if abs(upstream - downstream) >= 25:
            reason_parts.append(f"上下游交通量差 {abs(upstream-downstream):.0f}，需关注排队传播")
        if weather not in {"Clear", "晴", "未知", ""}:
            reason_parts.append(f"天气为{weather}")

        return {
            "engine": "policy-safety-fallback",
            "engine_note": "依据冻结政策库和安全门生成；未调用外部模型。",
            "operational_state": "进入人工复核" if score >= 0.6 else "保留观察",
            "type_hypothesis": type_name,
            "type_is_hypothesis": True,
            "severity": "待人工确认",
            "human_review_required": True,
            "reasoning": "；".join(reason_parts) + "。",
            "actions": [
                {"code": ACTION_CODES[name], "label": name, "authorized": True}
                for name in actions
                if name in ACTION_CODES
            ],
            "unsupported_actions": [],
            "policy_version": self.policy_library.get("policy_version", "1.0"),
        }

    def _call_ollama(self, packet: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        allowed = [item["label"] for item in fallback["actions"]]
        prompt = (
            "你是高速公路风险复核助手。只能从授权动作中选择，不得判断事故已经发生，"
            "不得省略人工复核。请仅返回JSON，字段为 operational_state、reasoning、actions、"
            "human_review_required。\n"
            f"证据包：{json.dumps(packet, ensure_ascii=False)}\n"
            f"授权动作：{json.dumps(allowed, ensure_ascii=False)}"
        )
        body = json.dumps(
            {"model": self.ollama_model, "prompt": prompt, "stream": False, "format": "json"},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(str(exc)) from exc
        return json.loads(envelope.get("response", "{}"))

    def _validate_llm(self, candidate: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        allowed_by_label = {item["label"]: item for item in fallback["actions"]}
        requested = candidate.get("actions", [])
        if not isinstance(requested, list):
            requested = []
        selected = []
        unsupported = []
        for value in requested:
            label = value.get("label", "") if isinstance(value, dict) else str(value)
            if label in allowed_by_label:
                selected.append(allowed_by_label[label])
            elif label:
                unsupported.append(label)
        if not selected:
            selected = fallback["actions"]
        return {
            **fallback,
            "engine": f"ollama:{self.ollama_model}",
            "engine_note": "本地大模型输出已通过字段、动作白名单和人工复核门禁。",
            "operational_state": str(candidate.get("operational_state", fallback["operational_state"])),
            "reasoning": str(candidate.get("reasoning", fallback["reasoning"])),
            "actions": selected,
            "unsupported_actions": unsupported,
            "human_review_required": True,
        }

