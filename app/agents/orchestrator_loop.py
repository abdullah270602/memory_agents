from __future__ import annotations
import json
from typing import Dict, List, Tuple, Any
import logging

from app.agents.base import Agent, DEFAULT_AGENTS
from app.agents.prompts import SYSTEM_TEMPLATE, CONTEXT_TEMPLATE
from app.memory.embedding import cosine_similarity
from app.memory.schemas import MemoryItem


class Orchestrator:
    def __init__(self, store: Any, llm: Any, embed_enabled: bool = True):
        self.store = store
        self.llm = llm
        self.embed_enabled = embed_enabled
        self.logger = logging.getLogger(__name__)

    def _format_short_context(self, items: List[MemoryItem]) -> str:
        lines = []
        for it in items:
            who = it.user_id if it.user_id else it.agent_id
            lines.append(f"[{it.timestamp}] {who}: {it.content}")
        return "\n".join(lines[-10:])

    def _rank_relevant(self, user_query: str, items: List[MemoryItem], top_k: int = 5) -> List[MemoryItem]:
        if not items:
            return []
        if not self.embed_enabled:
            items_sorted = sorted(items, key=lambda m: m.timestamp, reverse=True)
            return items_sorted[:top_k]
        query_emb = self.llm.embed(user_query)
        scored = []
        for it in items:
            if not it.embedding:
                continue
            score = cosine_similarity(query_emb, it.embedding)
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]

    def _build_messages(
        self,
        agent: Agent,
        short_context: List[MemoryItem],
        personal_mem: List[MemoryItem],
        shared_mem: List[MemoryItem],
        user_message: str,
    ) -> List[Dict[str, str]]:
        sys = SYSTEM_TEMPLATE.format(name=agent.name, role=agent.role, persona=agent.persona)
        short_txt = self._format_short_context(short_context)
        personal_txt = "\n".join(f"- {m.content}" for m in personal_mem) or "(none)"
        shared_txt = "\n".join(f"- {m.content}" for m in shared_mem) or "(none)"
        ctx = CONTEXT_TEMPLATE.format(short_context=short_txt, personal_memory=personal_txt, shared_memory=shared_txt)
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": ctx + f"\nUser says: {user_message}\nRespond with JSON only."},
        ]

    def _normalize_type(self, t: str) -> str:
        if not t:
            return "personal"
        l = t.lower()
        if "person" in l:
            return "personal"
        if "long" in l or "shared" in l:
            return "long_term"
        return "personal"

    def run_agents(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        agent_ids: List[str],
    ) -> Tuple[List[Dict[str, str]], List[Dict], Dict]:
        replies: List[Dict[str, str]] = []
        mem_changes: List[Dict] = []
        trace: Dict = {"agents": []}

        short_context = self.store.get_short_term(conversation_id, limit=20)

        for aid in agent_ids:
            agent = DEFAULT_AGENTS.get(aid)
            if agent is None:
                continue

            # Personal long-term
            pkeys = self.store.get_long_term_personal_keys(agent_id=agent.id, user_id=user_id, limit=200)
            personal_items = self.store.get_by_keys(pkeys)
            personal_ranked = self._rank_relevant(user_message, personal_items, top_k=5)

            # Shared long-term (per conversation)
            skeys = self.store.get_long_term_shared_keys(conversation_id=conversation_id, limit=200)
            shared_items = self.store.get_by_keys(skeys)
            shared_ranked = self._rank_relevant(user_message, shared_items, top_k=5)

            self.logger.info(
                "orchestrate agent=%s short=%d personal=%d shared=%d",
                agent.id, len(short_context), len(personal_ranked), len(shared_ranked)
            )
            messages = self._build_messages(agent, short_context, personal_ranked, shared_ranked, user_message)
            raw = self.llm.chat(messages)
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"reply": raw, "memories_to_write": [], "delegate_to": [], "done": False}

            reply_txt = str(parsed.get("reply", ""))
            replies.append({"agent_id": agent.id, "message": reply_txt})

            # Save agent reply to short-term
            self.store.save_memory(
                agent_id=agent.id,
                user_id="",  # agent
                conversation_id=conversation_id,
                type="short_term",
                content=reply_txt,
            )

            # Apply memory writes
            for mem in parsed.get("memories_to_write", []) or []:
                t_in = mem.get("type", "personal")
                t = self._normalize_type(t_in)
                content = str(mem.get("content", "")).strip()
                target = mem.get("target", "personal")  # personal|shared|none
                if not content:
                    continue
                target_user_id = user_id if target == "personal" else ""
                target_conv = conversation_id

                emb = None
                if self.embed_enabled and t in ("long_term", "personal"):
                    emb = self.llm.embed(content)

                saved = self.store.save_memory(
                    agent_id=agent.id,
                    user_id=target_user_id,
                    conversation_id=target_conv,
                    type=t,
                    content=content,
                    embedding=emb,
                )
                self.logger.info(
                    "mem_auto_write agent=%s type=%s key=%s target=%s", agent.id, t, saved.key, target
                )
                mem_changes.append({"op": "add", "key": saved.key, "type": t})

            trace["agents"].append({
                "agent_id": agent.id,
                "short_count": len(short_context),
                "personal_keys": [m.key for m in personal_ranked],
                "shared_keys": [m.key for m in shared_ranked],
            })

        return replies, mem_changes, trace

    def run_loop(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        agent_ids: List[str],
        max_steps: int = 3,
    ) -> Tuple[List[Dict[str, str]], List[Dict], Dict]:
        replies: List[Dict[str, str]] = []
        mem_changes: List[Dict] = []
        steps: List[Dict] = []

        queue = list(agent_ids)
        step = 0
        done_any = False
        while queue and step < max_steps and not done_any:
            current = queue
            queue = []
            step_trace = {"step": step + 1, "agents": []}
            for aid in current:
                r, m, tr = self.run_agents(conversation_id, user_id, user_message, [aid])
                replies.extend(r)
                mem_changes.extend(m)
                step_trace["agents"].extend(tr.get("agents", []))

                # Try to parse control hints from the agent reply if it is JSON
                delegates: List[str] = []
                done_flag = False
                try:
                    last_msg = r[-1]["message"] if r else ""
                    maybe = json.loads(last_msg) if isinstance(last_msg, str) and last_msg.strip().startswith("{") else {}
                    delegates = [d for d in maybe.get("delegate_to", []) if d in DEFAULT_AGENTS]
                    done_flag = bool(maybe.get("done", False))
                except Exception:
                    done_flag = False
                if done_flag:
                    done_any = True
                for d in delegates:
                    if d not in queue and d in DEFAULT_AGENTS:
                        queue.append(d)
            steps.append(step_trace)
            step += 1

        return replies, mem_changes, {"steps": steps}

