from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from rag_app.config import Settings


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _token_counts_from_completion_usage(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    if isinstance(usage, Mapping):
        prompt = usage.get("prompt_tokens")
        if prompt is None:
            prompt = usage.get("input_tokens")
        completion = usage.get("completion_tokens")
        if completion is None:
            completion = usage.get("output_tokens")
        return int(prompt or 0), int(completion or 0)
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", None)
    return int(prompt or 0), int(completion or 0)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: TokenUsage


class LLMClient:
    def __init__(self, settings: Settings):
        self.model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def generate_query_rewrites(self, question: str, count: int) -> tuple[list[str], TokenUsage]:
        if count <= 0:
            return [], TokenUsage()
        system = (
            "Ты генерируешь поисковые запросы для RAG по русскоязычной документации "
            "медицинской информационной системы. Верни только JSON."
        )
        user = (
            "Создай разные варианты поискового запроса для вопроса пользователя. "
            "Добавь вероятные термины интерфейса, синонимы и аббревиатуры, если это полезно. "
            "Не отвечай на вопрос.\n\n"
            f"Вопрос: {question}\n\n"
            f"Верни JSON строго в таком формате: {{\"queries\": [\"...\", до {count} строк]}}"
        )
        try:
            result = await self._chat(system, user, temperature=0.2, max_tokens=450)
            data = _parse_json_object(result.content)
            queries = [str(item).strip() for item in data.get("queries", []) if str(item).strip()]
        except Exception:
            return [], TokenUsage()
        return _unique(queries)[:count], result.usage

    async def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, float]], TokenUsage]:
        compact_candidates = [
            {
                "chunk_id": candidate["chunk_id"],
                "source": candidate["source_ref"],
                "text": _truncate(candidate["text"], 1400),
            }
            for candidate in candidates
        ]
        system = (
            "Ты строгий reranker релевантности для RAG по документации медицинской информационной системы. "
            "Ранжируй фрагменты только по тому, помогают ли они ответить на вопрос пользователя. "
            "Верни только JSON."
        )
        user = (
            f"Вопрос: {question}\n\n"
            f"Кандидатные фрагменты:\n{json.dumps(compact_candidates, ensure_ascii=False)}\n\n"
            "Верни JSON строго в таком формате, отсортированный от лучшего к худшему: "
            "{\"ranking\": [{\"chunk_id\": 123, \"score\": 0-100}]}. "
            "Для нерелевантных фрагментов (не помогают ответить на вопрос) используй score -1."
        )
        try:
            chat_result = await self._chat(system, user, temperature=0.0, max_tokens=900)
            data = _parse_json_object(chat_result.content)
            ranking = data.get("ranking", [])
            result: list[tuple[int, float]] = []
            for item in ranking:
                chunk_id = int(item["chunk_id"])
                score = float(item.get("score", 0))
                if score < 0:
                    result.append((chunk_id, -1.0))
                else:
                    result.append((chunk_id, max(0.0, min(100.0, score))))
            return result, chat_result.usage
        except Exception:
            return [], TokenUsage()

    async def answer(self, question: str, contexts: list[dict[str, Any]]) -> ChatResult:
        context_text = "\n\n".join(
            (
                f"[{context['source_number']}] {context['source_ref']}\n"
                f"{_truncate(context['text'], 2200)}"
            )
            for context in contexts
        )
        system = (
            "Ты полезный RAG-ассистент по документации медицинской информационной системы. "
            "Отвечай на языке пользователя. Используй только предоставленный контекст. "
            "Если ответа нет в контексте, скажи, что в документации недостаточно информации. "
            "Указывай источники внутри ответа в формате [1], [2] и так далее рядом с утверждениями. "
            "Не добавляй в конце раздел «Источники:» — он формируется автоматически."
        )
        user = (
            f"Вопрос:\n{question}\n\n"
            f"Контекст:\n{context_text}\n\n"
            "Напиши практичный и краткий ответ. Ставь ссылки на источники рядом с соответствующими утверждениями."
        )
        return await self._chat(system, user, temperature=self._temperature, max_tokens=1800)

    async def _chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        inp, out = _token_counts_from_completion_usage(response.usage)
        return ChatResult(
            content=response.choices[0].message.content or "",
            usage=TokenUsage(input_tokens=inp, output_tokens=out),
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."
