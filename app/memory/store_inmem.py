from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from app.memory.schemas import MemoryItem
import logging


logger = logging.getLogger(__name__)


class InMemoryStore:
    def __init__(self):
        self.mem: Dict[str, MemoryItem] = {}
        self.idx_short: Dict[str, List[str]] = {}
        self.idx_lt_personal: Dict[str, set] = {}
        self.idx_lt_shared: Dict[str, set] = {}

    def _mem_key(self) -> str:
        return f"mem:{uuid.uuid4().hex}"

    def _lt_p_key(self, agent_id: str, user_id: str) -> str:
        return f"{agent_id}:{user_id}"

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
        k = key or self._mem_key()
        item = MemoryItem(
            key=k,
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
            type=type,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            embedding=embedding,
        )
        self.mem[k] = item
        if type == "short_term":
            self.idx_short.setdefault(conversation_id, []).append(k)
        elif type in ("long_term", "personal"):
            if user_id:
                self.idx_lt_personal.setdefault(self._lt_p_key(agent_id, user_id), set()).add(k)
            else:
                self.idx_lt_shared.setdefault(conversation_id, set()).add(k)
        logger.info("mem_save type=%s key=%s len=%d", type, k, len(content))
        return item

    def edit_memory(self, key: str, new_content: str) -> Optional[MemoryItem]:
        item = self.mem.get(key)
        if not item:
            return None
        item.content = new_content
        item.timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("mem_edit key=%s", key)
        return item

    def delete_memory(self, key: str) -> bool:
        item = self.mem.pop(key, None)
        if not item:
            return False
        if item.type == "short_term":
            lst = self.idx_short.get(item.conversation_id, [])
            self.idx_short[item.conversation_id] = [k for k in lst if k != key]
        elif item.type in ("long_term", "personal"):
            if item.user_id:
                p = self._lt_p_key(item.agent_id, item.user_id)
                self.idx_lt_personal.get(p, set()).discard(key)
            else:
                self.idx_lt_shared.get(item.conversation_id, set()).discard(key)
        logger.info("mem_delete key=%s", key)
        return True

    def get_short_term(self, conversation_id: str, limit: int = 20) -> List[MemoryItem]:
        keys = self.idx_short.get(conversation_id, [])[-limit:]
        return [self.mem[k] for k in keys if k in self.mem]

    def get_long_term_personal_keys(self, agent_id: str, user_id: str, limit: int = 200) -> List[str]:
        p = self._lt_p_key(agent_id, user_id)
        keys = list(self.idx_lt_personal.get(p, set()))
        return keys[:limit]

    def get_long_term_shared_keys(self, conversation_id: str, limit: int = 200) -> List[str]:
        keys = list(self.idx_lt_shared.get(conversation_id, set()))
        return keys[:limit]

    def get_by_keys(self, keys: Iterable[str]) -> List[MemoryItem]:
        return [self.mem[k] for k in keys if k in self.mem]

