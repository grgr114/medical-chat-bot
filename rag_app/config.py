from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_BASE_URL = "http://127.0.0.1:1488/v1"
DEFAULT_MODEL = "lmstudio-community/gemma-4-e4b-it"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Medical RAG"
    chunks_path: str = "pipeline/output/chunks.jsonl"
    texts_csv_path: str = "pipeline/texts.csv"
    auto_index: bool = False
    force_reindex_on_start: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "medical_chunks"

    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"

    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_device: str | None = None

    @field_validator("embedding_device", mode="before")
    @classmethod
    def empty_embedding_device_means_auto(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return str(value).strip() if isinstance(value, str) else value
    embedding_batch_size: int = 16
    # Jina v3 uses encode(..., task=...) for asymmetric retrieval; leave prefixes empty.
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    # Set to empty to omit task= for models that do not support it.
    embedding_query_task: str = "retrieval.query"
    embedding_passage_task: str = "retrieval.passage"

    llm_base_url: str = Field(default=DEFAULT_BASE_URL, validation_alias="LLM_BASE_URL")
    llm_model: str = Field(default=DEFAULT_MODEL, validation_alias="LLM_MODEL")
    llm_api_key: str = "lm-studio"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 120.0

    dense_limit: int = 24
    sparse_limit: int = 24
    candidate_limit: int = 24
    answer_context_limit: int = 6
    rrf_k: int = 60
    query_rewrite_count: int = 3
    enable_query_rewrite: bool = True
    enable_llm_rerank: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()

