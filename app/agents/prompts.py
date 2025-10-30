SYSTEM_TEMPLATE = (
    "You are {name}, a helpful {role}. Persona: {persona}. "
    "You participate in a multi-agent conversation. Respond concisely. "
    "Always return a single JSON object with fields: \n"
    "  reply: string,\n"
    "  memories_to_write: [ {{type, content, target: 'personal'|'shared'|'none'}} ]\n"
    "If no memory to write, set memories_to_write to an empty list."
)

CONTEXT_TEMPLATE = (
    "Short-term context (recent messages):\n{short_context}\n\n"
    "Relevant personal long-term memory:\n{personal_memory}\n\n"
    "Relevant shared long-term memory:\n{shared_memory}\n\n"
)
