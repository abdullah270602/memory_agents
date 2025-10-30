from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


class MemoryItem(BaseModel):
    key: str
    agent_id: str
    user_id: str
    conversation_id: str
    type: str  # short_term | long_term | personal
    content: str
    timestamp: str
    embedding: Optional[List[float]] = None


class MemoryOp(BaseModel):
    op: str  # add|edit|delete
    type: str  # short_term|long_term|personal
    key: Optional[str] = None
    content: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    target: Optional[str] = None  # personal|shared (for add)
