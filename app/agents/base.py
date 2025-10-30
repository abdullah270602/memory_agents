from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Agent:
    id: str
    name: str
    role: str
    persona: str


DEFAULT_AGENTS = {
    "assistant": Agent(
        id="assistant",
        name="Assistant",
        role="general assistant",
        persona="Helpful, concise, friendly."
    ),
    "planner": Agent(
        id="planner",
        name="Planner",
        role="planner",
        persona="Turns goals into stepwise plans; pragmatic and clear."
    ),
    "critic": Agent(
        id="critic",
        name="Critic",
        role="reviewer",
        persona="Reviews output for issues and suggests improvements."
    ),
}

