from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
from openai import OpenAI
from app.config import get_settings


class LLMClient:
    def __init__(self):
        s = get_settings()
        self.logger = logging.getLogger(__name__)
        headers = {
            "HTTP-Referer": "mem-agents.local",
            "X-Title": "MemGPT-Style Multi-Agent System",
        }
        self.client = OpenAI(
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
            default_headers=headers,
        )
        self.chat_model = s.openai_model
        self.embedding_model = s.embedding_model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        self.logger.info(
            "llm_chat model=%s msgs=%d", self.chat_model, len(messages)
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self.logger.warning(
                "llm_chat_json_mode_unsupported model=%s err=%s -> retrying without response_format",
                self.chat_model,
                exc,
            )
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
            )
        content = resp.choices[0].message.content or "{}"
        return content

    def embed(self, text: str) -> List[float]:
        self.logger.info("llm_embed model=%s len=%d", self.embedding_model, len(text))
        try:
            e = self.client.embeddings.create(model=self.embedding_model, input=text)
            return list(e.data[0].embedding)
        except Exception as exc:
            # Fallback to deterministic local embedding when remote model is unavailable
            self.logger.warning("llm_embed_fallback reason=%s", exc)
            vec = [0.0] * 64
            b = text.encode("utf-8", errors="ignore")
            for i, by in enumerate(b):
                vec[i % 64] += (by / 255.0)
            norm = sum(v * v for v in vec) ** 0.5 or 1e-9
            return [v / norm for v in vec]
