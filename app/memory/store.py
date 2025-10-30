from __future__ import annotations
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import redis

from app.memory.schemas import MemoryItem


logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, redis_url: str):
        if not redis_url:
            raise ValueError("UPSTASH_REDIS_URL not configured")
        self.r = redis.from_url(redis_url, decode_responses=True, ssl=True)

    # Key helpers
    def _mem_key(self, mid: Optional[str] = None) -> str:
        return f"mem:{mid or uuid.uuid4().hex}"

    def _idx_short(self, conversation_id: str) -> str:
        return f"idx:short:{conversation_id}"

    def _idx_lt_personal(self, agent_id: str, user_id: str) -> str:
        return f"idx:lt:{agent_id}:{user_id}"

    def _idx_lt_shared(self, conversation_id: str) -> str:
        return f"idx:lt_shared:{conversation_id}"

    # Core ops
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
        self.r.set(mkey, item.model_dump_json())

        if type == "short_term":
            # Use a list to preserve order
            self.r.rpush(self._idx_short(conversation_id), mkey)
        elif type in ("long_term", "personal"):
            if user_id:
                self.r.sadd(self._idx_lt_personal(agent_id, user_id), mkey)
            else:
                self.r.sadd(self._idx_lt_shared(conversation_id), mkey)
        logger.info("mem_save type=%s key=%s len=%d", type, mkey, len(content))
        return item

    def edit_memory(self, key: str, new_content: str) -> Optional[MemoryItem]:
        data = self.r.get(key)
        if not data:
            return None
        obj = MemoryItem.model_validate_json(data)
        obj.content = new_content
        obj.timestamp = datetime.now(timezone.utc).isoformat()
        self.r.set(key, obj.model_dump_json())
        logger.info("mem_edit key=%s", key)
        return obj

    def delete_memory(self, key: str) -> bool:
        data = self.r.get(key)
        if not data:
            return False
        obj = MemoryItem.model_validate_json(data)
        self.r.delete(key)
        if obj.type == "short_term":
            self.r.lrem(self._idx_short(obj.conversation_id), 0, key)
        elif obj.type in ("long_term", "personal"):
            if obj.user_id:
                self.r.srem(self._idx_lt_personal(obj.agent_id, obj.user_id), key)
            else:
                self.r.srem(self._idx_lt_shared(obj.conversation_id), key)
        logger.info("mem_delete key=%s", key)
        return True

    # Retrieval
    def get_short_term(self, conversation_id: str, limit: int = 20) -> List[MemoryItem]:
        idx = self._idx_short(conversation_id)
        keys = self.r.lrange(idx, max(0, -limit), -1)
        out: List[MemoryItem] = []
        pipe = self.r.pipeline()
        for k in keys:
            pipe.get(k)
        vals = pipe.execute()
        for v in vals:
            if v:
                out.append(MemoryItem.model_validate_json(v))
        return out

    def get_long_term_personal_keys(self, agent_id: str, user_id: str, limit: int = 200) -> List[str]:
        idx = self._idx_lt_personal(agent_id, user_id)
        keys = list(self.r.smembers(idx))
        return keys[:limit]

    def get_long_term_shared_keys(self, conversation_id: str, limit: int = 200) -> List[str]:
        idx = self._idx_lt_shared(conversation_id)
        keys = list(self.r.smembers(idx))
        return keys[:limit]

    def get_by_keys(self, keys: Iterable[str]) -> List[MemoryItem]:
        keys_list = list(keys)
        if not keys_list:
            return []
        pipe = self.r.pipeline()
        for k in keys_list:
            pipe.get(k)
        vals = pipe.execute()
        out: List[MemoryItem] = []
        for v in vals:
            if v:
                out.append(MemoryItem.model_validate_json(v))
        return out
