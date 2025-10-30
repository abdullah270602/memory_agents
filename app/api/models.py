from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.memory.schemas import MemoryOp


class ChatRequest(BaseModel):
    # Minimal API: message + user_id or username
    message: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    # Optional overrides
    conversation_id: Optional[str] = None
    agent_ids: Optional[List[str]] = None
    memory_ops: Optional[List[MemoryOp]] = None
    describe: Optional[bool] = None


class ChatResponse(BaseModel):
    replies: List[Dict[str, str]]
    memory_changes: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None
