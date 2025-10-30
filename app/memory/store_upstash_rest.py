from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import httpx
import logging

from app.memory.schemas import MemoryItem


logger = logging.getLogger(__name__)


class UpstashRest:
    def __init__(self, base_url: str, token: str):
        if not base_url or not token:
            raise ValueError("Upstash REST URL/token missing")
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def pipeline(self, commands: List[List[str]]):
        url = f"{self.base_url}/pipeline"
        with httpx.Client(timeout=15) as client:
            r = client.post(url, headers=self.headers, json={"commands": commands})
            r.raise_for_status()
            return r.json()

    # Convenience ops
    def set(self, key: str, value: str):
        return self.pipeline([["SET", key, value]])

    def get(self, key: str):
        res = self.pipeline([["GET", key]])
        return res[0][1]

    def delete(self, key: str):
        return self.pipeline([["DEL", key]])

    def rpush(self, key: str, value: str):
        return self.pipeline([["RPUSH", key, value]])

    def lrange(self, key: str, start: int, end: int):
        res = self.pipeline([["LRANGE", key, str(start), str(end)]])
        return res[0][1] or []

    def lrem(self, key: str, count: int, value: str):
        return self.pipeline([["LREM", key, str(count), value]])

    def sadd(self, key: str, value: str):
        return self.pipeline([["SADD", key, value]])

    def smembers(self, key: str):
        res = self.pipeline([["SMEMBERS", key]])
        return res[0][1] or []

    def srem(self, key: str, value: str):
        return self.pipeline([["SREM", key, value]])


class UpstashRestStore:
    def __init__(self, base_url: str, token: str):
        self.redis = UpstashRest(base_url, token)

    def _mem_key(self) -> str:
        return f"mem:{uuid.uuid4().hex}"

    def _idx_short(self, conversation_id: str) -> str:
        return f"idx:short:{conversation_id}"

    def _idx_lt_personal(self, agent_id: str, user_id: str) -> str:
        return f"idx:lt:{agent_id}:{user_id}"

    def _idx_lt_shared(self, conversation_id: str) -> str:
        return f"idx:lt_shared:{conversation_id}"

    def save_memory(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: str,
        type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        key: Optional[str] = None,
    ) -> MemoryItem:
        now = datetime.now(timezone.utc).isoformat()
        mkey = key or self._mem_key()
        item = MemoryItem(
            key=mkey,
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
            type=type,
            content=content,
            timestamp=now,
            embedding=embedding,
        )
        self.redis.set(mkey, item.model_dump_json())
        if type == "short_term":
            self.redis.rpush(self._idx_short(conversation_id), mkey)
        elif type in ("long_term", "personal"):
            if user_id:
                self.redis.sadd(self._idx_lt_personal(agent_id, user_id), mkey)
            else:
                self.redis.sadd(self._idx_lt_shared(conversation_id), mkey)
        logger.info("mem_save type=%s key=%s len=%d", type, mkey, len(content))
        return item

    def edit_memory(self, key: str, new_content: str) -> Optional[MemoryItem]:
        data = self.redis.get(key)
        if data is None:
            return None
        obj = MemoryItem.model_validate_json(data)
        obj.content = new_content
        obj.timestamp = datetime.now(timezone.utc).isoformat()
        self.redis.set(key, obj.model_dump_json())
        logger.info("mem_edit key=%s", key)
        return obj

    def delete_memory(self, key: str) -> bool:
        data = self.redis.get(key)
        if data is None:
            return False
        obj = MemoryItem.model_validate_json(data)
        self.redis.delete(key)
        if obj.type == "short_term":
            self.redis.lrem(self._idx_short(obj.conversation_id), 0, key)
        elif obj.type in ("long_term", "personal"):
            if obj.user_id:
                self.redis.srem(self._idx_lt_personal(obj.agent_id, obj.user_id), key)
            else:
                self.redis.srem(self._idx_lt_shared(obj.conversation_id), key)
        logger.info("mem_delete key=%s", key)
        return True

    def get_short_term(self, conversation_id: str, limit: int = 20) -> List[MemoryItem]:
        idx = self._idx_short(conversation_id)
        keys = self.redis.lrange(idx, max(0, -limit), -1)
        out: List[MemoryItem] = []
        if not keys:
            return out
        # Pipeline GETs
        res = self.redis.pipeline([["GET", k] for k in keys])
        for r in res:
            v = r[1]
            if v:
                out.append(MemoryItem.model_validate_json(v))
        return out

    def get_long_term_personal_keys(self, agent_id: str, user_id: str, limit: int = 200) -> List[str]:
        idx = self._idx_lt_personal(agent_id, user_id)
        keys = self.redis.smembers(idx)
        return list(keys)[:limit]

    def get_long_term_shared_keys(self, conversation_id: str, limit: int = 200) -> List[str]:
        idx = self._idx_lt_shared(conversation_id)
        keys = self.redis.smembers(idx)
        return list(keys)[:limit]

    def get_by_keys(self, keys: Iterable[str]) -> List[MemoryItem]:
        keys_list = list(keys)
        if not keys_list:
            return []
        res = self.redis.pipeline([["GET", k] for k in keys_list])
        out: List[MemoryItem] = []
        for r in res:
            v = r[1]
            if v:
                out.append(MemoryItem.model_validate_json(v))
        return out

