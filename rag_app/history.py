from __future__ import annotations

import json
import asyncio
import uuid
from typing import Any
from uuid import UUID

import asyncpg

from rag_app.config import Settings


class HistoryStore:
    def __init__(self, settings: Settings):
        self._dsn = settings.postgres_dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self, attempts: int = 30, delay_seconds: float = 1.0) -> None:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
                await self.init_schema()
                return
            except Exception as exc:  # Postgres may still be starting under docker-compose.
                last_error = exc
                await asyncio.sleep(delay_seconds)
        raise RuntimeError("PostgreSQL did not become ready") from last_error

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def init_schema(self) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_sessions (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS rag_calls (
                    id UUID PRIMARY KEY,
                    session_id UUID NOT NULL REFERENCES rag_sessions(id) ON DELETE CASCADE,
                    user_message TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources JSONB NOT NULL,
                    model TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_rag_calls_session_created
                    ON rag_calls (session_id, created_at);
                """
            )
            await conn.execute(
                """
                ALTER TABLE rag_calls
                ADD COLUMN IF NOT EXISTS input_tokens INTEGER NOT NULL DEFAULT 0;

                ALTER TABLE rag_calls
                ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0;
                """
            )

    async def ensure_session(self, session_id: UUID | None, first_message: str) -> UUID:
        pool = self._require_pool()
        session_uuid = session_id or uuid.uuid4()
        title = _make_title(first_message)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO rag_sessions (id, title)
                VALUES ($1, $2)
                ON CONFLICT (id) DO UPDATE SET updated_at = now()
                """,
                session_uuid,
                title,
            )
        return session_uuid

    async def insert_call(
        self,
        *,
        session_id: UUID,
        user_message: str,
        answer: str,
        sources: list[dict[str, Any]],
        model: str,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> UUID:
        pool = self._require_pool()
        call_id = uuid.uuid4()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO rag_calls (
                        id,
                        session_id,
                        user_message,
                        answer,
                        sources,
                        model,
                        latency_ms,
                        input_tokens,
                        output_tokens
                    )
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9)
                    """,
                    call_id,
                    session_id,
                    user_message,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    model,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                )
                await conn.execute(
                    "UPDATE rag_sessions SET updated_at = now() WHERE id = $1",
                    session_id,
                )
        return call_id

    async def list_sessions(self, limit: int = 25) -> list[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, created_at, updated_at
                FROM rag_sessions
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(row) for row in rows]

    async def delete_session(self, session_id: UUID) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM rag_sessions WHERE id = $1",
                session_id,
            )
        return result == "DELETE 1"

    async def list_calls(self, session_id: UUID) -> list[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    session_id,
                    user_message,
                    answer,
                    sources,
                    model,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    created_at
                FROM rag_calls
                WHERE session_id = $1
                ORDER BY created_at ASC
                """,
                session_id,
            )
        calls: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if isinstance(item["sources"], str):
                item["sources"] = json.loads(item["sources"])
            calls.append(item)
        return calls

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("HistoryStore is not connected")
        return self._pool


def _make_title(message: str) -> str:
    title = " ".join(message.split())
    if len(title) > 80:
        return title[:79].rstrip() + "..."
    return title or "New chat"
