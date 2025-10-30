from __future__ import annotations
import json
from typing import List
import logging
import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.models import ChatRequest, ChatResponse
from app.config import get_settings
from app.memory.schemas import MemoryOp
from app.agents.orchestrator import Orchestrator
from app.utils.logging_config import setup_logging, request_id_ctx

setup_logging()
logger = logging.getLogger("app")
app = FastAPI(title="MemGPT-Style Multi-Agent System")


settings = get_settings()
MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"  # legacy combined switch

# Decide store backend
store_backend = (settings.store_backend or ("inmem" if MOCK_MODE else "")).lower()
llm_backend = (settings.llm_backend or ("dummy" if MOCK_MODE else "")).lower()

if store_backend == "inmem":
    logger.info("startup store=inmem (env override)")
    from app.memory.store_inmem import InMemoryStore
    store = InMemoryStore()
elif store_backend == "redis":
    logger.info("startup store=redis (env override)")
    if not settings.upstash_redis_url:
        raise RuntimeError("STORE_BACKEND=redis but UPSTASH_REDIS_URL not set")
    from app.memory.store import MemoryStore
    store = MemoryStore(settings.upstash_redis_url)
elif store_backend == "upstash_rest":
    logger.info("startup store=upstash_rest (env override)")
    if not (settings.upstash_rest_url and settings.upstash_rest_token):
        raise RuntimeError("STORE_BACKEND=upstash_rest but REST URL/TOKEN not set")
    from app.memory.store_upstash_rest import UpstashRestStore
    from app.memory.store_inmem import InMemoryStore
    from app.memory.facade import StoreFacade
    primary = UpstashRestStore(settings.upstash_rest_url, settings.upstash_rest_token)
    store = StoreFacade(primary=primary, fallback=InMemoryStore())
else:
    # Auto choose
    if settings.upstash_redis_url:
        logger.info("startup store=redis (auto)")
        from app.memory.store import MemoryStore
        store = MemoryStore(settings.upstash_redis_url)
    elif settings.upstash_rest_url and settings.upstash_rest_token:
        logger.info("startup store=upstash_rest (auto)")
        from app.memory.store_upstash_rest import UpstashRestStore
        from app.memory.store_inmem import InMemoryStore
        from app.memory.facade import StoreFacade
        primary = UpstashRestStore(settings.upstash_rest_url, settings.upstash_rest_token)
        store = StoreFacade(primary=primary, fallback=InMemoryStore())
    else:
        logger.info("startup store=inmem (auto)")
        from app.memory.store_inmem import InMemoryStore
        store = InMemoryStore()

# Decide LLM backend
if llm_backend == "dummy":
    logger.info("startup llm=dummy (env override)")
    from app.llm.dummy import DummyLLMClient
    llm = DummyLLMClient()
elif llm_backend == "openrouter":
    logger.info("startup llm=openrouter (env override)")
    if not settings.openai_api_key:
        raise RuntimeError("LLM_BACKEND=openrouter but OPENAI_API_KEY not set")
    from app.llm.client import LLMClient
    llm = LLMClient()
else:
    # Auto choose
    if settings.openai_api_key:
        logger.info("startup llm=openrouter (auto)")
        from app.llm.client import LLMClient
        llm = LLMClient()
    else:
        logger.info("startup llm=dummy (auto)")
        from app.llm.dummy import DummyLLMClient
        llm = DummyLLMClient()

orc = Orchestrator(store, llm, embed_enabled=settings.use_embeddings) if store else None


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_ctx.set(rid)
    t0 = time.time()
    logger.info("request_start method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
        return response
    finally:
        dt = int((time.time() - t0) * 1000)
        status = getattr(locals().get("response", None), "status_code", "?")
        logger.info("request_end method=%s path=%s status=%s ms=%d", request.method, request.url.path, status, dt)
        request_id_ctx.reset(token)


@app.get("/health")
def health():
    ok = bool(store and orc)
    store_kind = "unknown"
    try:
        from app.memory.facade import StoreFacade as _SF
        if isinstance(store, _SF):
            # Inspect inner stores to report status
            active = store.active_backend
            kind = "upstash_rest" if active == "primary" else "inmem"
            store_kind = f"{kind}{' (degraded)' if active != 'primary' else ''}"
        else:
            from app.memory.store_inmem import InMemoryStore as _IM
            if isinstance(store, _IM):
                store_kind = "inmem"
            else:
                # assume redis
                store_kind = "redis"
    except Exception:
        store_kind = "unknown"
    try:
        from app.llm.dummy import DummyLLMClient as _DL
        llm_kind = "dummy" if isinstance(llm, _DL) else "openrouter"
    except Exception:
        llm_kind = "unknown"
    return {"ok": ok, "mode": "mock" if MOCK_MODE else "live", "backends": {"store": store_kind, "llm": llm_kind}}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not store or not orc:
        raise HTTPException(status_code=500, detail="Storage not configured")

    # Resolve conversation and user identifiers (support minimal payload)
    conv_id = req.conversation_id
    if settings.single_channel_mode or not conv_id:
        conv_id = settings.default_conversation_id
    sender_id = req.user_id or getattr(req, 'username', None)
    if not sender_id:
        raise HTTPException(status_code=400, detail="user_id or username is required")

    # Agents
    agents = req.agent_ids or [a.strip() for a in (settings.default_agent_ids or "assistant").split(",") if a.strip()]

    logger.info("chat_request conv=%s user=%s agents=%s", conv_id, sender_id, ",".join(agents))
    # Apply any programmatic memory ops first
    mem_changes = []
    for op in req.memory_ops or []:
        if op.op == "add":
            # infer defaults if missing
            t = op.type
            agent_id = op.agent_id or "assistant"
            # Decide target scope
            target = (op.target or ("personal" if t == "personal" else "shared")).lower()
            if target not in ("personal", "shared"):
                target = "personal"
            user_id = (op.user_id or sender_id) if target == "personal" else ""
            conv = op.conversation_id or conv_id
            content = op.content or ""
            emb = None
            if settings.use_embeddings and t in ("long_term", "personal") and content:
                emb = llm.embed(content)
            saved = store.save_memory(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conv,
                type=t,
                content=content,
                embedding=emb,
            )
            logger.info("op_add type=%s key=%s", t, saved.key)
            mem_changes.append({"op": "add", "key": saved.key, "type": t})
        elif op.op == "edit" and op.key and op.content is not None:
            updated = store.edit_memory(op.key, op.content)
            mem_changes.append({"op": "edit", "key": op.key, "ok": bool(updated)})
        elif op.op == "delete" and op.key:
            ok = store.delete_memory(op.key)
            mem_changes.append({"op": "delete", "key": op.key, "ok": ok})

    # Save the user message to short-term memory
    store.save_memory(
        agent_id="user",
        user_id=sender_id,
        conversation_id=conv_id,
        type="short_term",
        content=req.message,
    )

    replies, auto_changes, trace = orc.run_agents(
        conversation_id=conv_id,
        user_id=sender_id,
        user_message=req.message,
        agent_ids=agents,
    )
    mem_changes.extend(auto_changes)
    logger.info("chat_done replies=%d mem_changes=%d", len(replies), len(mem_changes))
    debug = None
    if getattr(req, 'describe', None):
        debug = {
            "conversation_id": conv_id,
            "user_id": sender_id,
            "agents": trace.get("agents", []),
            "writes": mem_changes,
        }
    return ChatResponse(replies=replies, memory_changes=mem_changes, debug=debug)
