import os
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    # Accept OPENAI_* or OPENROUTER_* for convenience
    openai_api_key: str = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "openrouter/auto")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    # Redis over TLS URL (rediss://) or Upstash REST (URL+TOKEN)
    upstash_redis_url: str = os.getenv("UPSTASH_REDIS_URL", "")
    upstash_rest_url: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
    upstash_rest_token: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    store_backend: str = os.getenv("STORE_BACKEND", "")  # "redis"|"inmem"|""
    llm_backend: str = os.getenv("LLM_BACKEND", "")      # "openrouter"|"dummy"|""
    # Conversation + agent defaults
    single_channel_mode: bool = os.getenv("SINGLE_CHANNEL_MODE", "1") == "1"
    default_conversation_id: str = os.getenv("DEFAULT_CONVERSATION_ID", "global")
    default_agent_ids: str = os.getenv("DEFAULT_AGENT_IDS", "assistant")
    # Retrieval/embeddings toggle
    use_embeddings: bool = os.getenv("USE_EMBEDDINGS", "0") == "1"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
