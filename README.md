## Memory Agents API (Discord Integration)

This service exposes a single HTTP API the Discord bot (or any frontend) can call to talk to the agents. With `SINGLE_CHANNEL_MODE=1` (default) every request is routed to the one global conversation, so you only need to pass the Discord user id and their message.

### Run the API
- Install deps: `pip install -r requirements.txt` (or use the existing `.venv`)
- Start server: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Default base URL: `http://localhost:8000`

Key env vars: `OPENAI_API_KEY` (or OPENROUTER), `DISCORD_BOT_TOKEN`, optional `USE_EMBEDDINGS=1`, `SINGLE_CHANNEL_MODE=1`, `DEFAULT_CONVERSATION_ID=global`, `DEFAULT_AGENT_IDS=assistant`.

### Endpoint: POST /chat
- **URL**: `POST /chat`
- **Purpose**: Send a user message, get replies from the configured agents.
- **Headers**: `Content-Type: application/json`
- **Request body**:
  - `message` (string, required): User’s text.
  - `user_id` (string, required if `username` absent): Unique user identifier (use Discord user id).
  - `username` (string, optional): Alternative to `user_id`.
  - `conversation_id` (string, optional): Ignored when `SINGLE_CHANNEL_MODE=1`; otherwise sets the thread/channel id.
  - `agent_ids` (array[string], optional): Agents to run. Defaults to `DEFAULT_AGENT_IDS` (comma-separated env var).
  - `memory_ops` (optional): Programmatic memory edits before the chat; shape matches `app/memory/schemas.py::MemoryOp` (`op` add|edit|delete, `type` short_term|long_term|personal, etc.).
  - `describe` (bool, optional): If true, the response includes a compact debug trace.

- **Sample request (single-chat Discord)**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "discord-user-123",
    "message": "Hey, what can you do?",
    "describe": true
  }'
```

- **Response shape** (`ChatResponse`):
  - `replies`: list of `{ "agent_id": "...", "message": "..." }`.
  - `memory_changes`: list of memory write/edit/delete operations applied during the run.
  - `debug` (only when `describe` is true): includes `conversation_id`, `user_id`, `agents` (context keys used), `writes` (memory changes).

Example response:
```json
{
  "replies": [
    { "agent_id": "assistant", "message": "Hi! I can reason over prior chat and memories." }
  ],
  "memory_changes": [
    { "op": "add", "key": "lt:assistant:global:...", "type": "short_term" }
  ],
  "debug": {
    "conversation_id": "global",
    "user_id": "discord-user-123",
    "agents": [{ "agent_id": "assistant", "short_count": 1, "personal_keys": [], "shared_keys": [] }],
    "writes": [{ "op": "add", "key": "lt:assistant:global:...", "type": "short_term" }]
  }
}
```

### Endpoint: GET /health
- **URL**: `GET /health`
- **Purpose**: Lightweight readiness probe; reports backend mode and whether storage/LLM are wired.

### How the Discord bot calls it
The provided bot in `discord_bot/bot.py` posts to `/chat` with:
```json
{
  "conversation_id": "<channel id>",  // ignored when SINGLE_CHANNEL_MODE=1
  "user_id": "<discord user id>",
  "message": "<message content>",
  "describe": true
}
```
Set `API_URL` (defaults to `http://localhost:8000/chat`) and `DISCORD_BOT_TOKEN` in your environment, then run `python discord_bot/bot.py`. Replies from all agents are forwarded back to the Discord channel/DM.
