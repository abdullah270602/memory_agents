from __future__ import annotations
import json
from typing import Dict, List


class DummyLLMClient:
    """Deterministic mock for local testing without network calls."""

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        # Take last user message content
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        last = user_parts[-1] if user_parts else ""
        # Simple heuristic: if text contains "remember:", write to personal memory
        mems = []
        lower = last.lower()
        if "remember:" in lower:
            idx = lower.find("remember:")
            content = last[idx + len("remember:") :].strip()[:500]
            if content:
                mems.append({"type": "personal", "content": content, "target": "personal"})
        reply = {
            "reply": f"[mock] Noted. You said: {last[-120:]}",
            "memories_to_write": mems,
        }
        return json.dumps(reply)

    def embed(self, text: str) -> List[float]:
        # 64-dim toy embedding based on bytes
        vec = [0.0] * 64
        b = text.encode("utf-8", errors="ignore")
        for i, by in enumerate(b):
            vec[i % 64] += (by / 255.0)
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5 or 1e-9
        return [v / norm for v in vec]
