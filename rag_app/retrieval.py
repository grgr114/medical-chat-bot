from __future__ import annotations

import logging
from dataclasses import dataclass

from pathlib import Path

from rag_app.bm25 import BM25Index
from rag_app.citations import strip_trailing_sources_block
from rag_app.config import Settings
from rag_app.documents import ChunkDocument
from rag_app.embeddings import EmbeddingModel
from rag_app.llm import LLMClient, TokenUsage
from rag_app.qdrant_store import QdrantStore
from rag_app.schemas import Source
from rag_app.wiki_urls import load_wiki_title_url_map, wiki_url_for_doc_title

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    doc: ChunkDocument
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    def to_source(self, doc_url: str | None = None) -> Source:
        return Source(
            chunk_id=self.doc.chunk_id,
            source_ref=self.doc.source_ref,
            doc_title=self.doc.doc_title,
            h1=self.doc.h1,
            h2=self.doc.h2,
            source_file=self.doc.source_file,
            snippet=self.doc.snippet(),
            doc_url=doc_url,
            dense_score=self.dense_score,
            sparse_score=self.sparse_score,
            fused_score=self.fused_score,
            rerank_score=self.rerank_score,
        )


class RAGService:
    def __init__(
        self,
        *,
        settings: Settings,
        docs: list[ChunkDocument],
        embeddings: EmbeddingModel,
        qdrant: QdrantStore,
        bm25: BM25Index,
        llm: LLMClient,
    ):
        self.settings = settings
        self.docs = docs
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.bm25 = bm25
        self.llm = llm
        self._wiki_urls = load_wiki_title_url_map(Path(settings.texts_csv_path))

    def build_index(self, force: bool = False) -> bool:
        self.qdrant.ensure_collection(vector_size=self.embeddings.dimension, force=force)
        try:
            existing_count = self.qdrant.count()
        except Exception:
            existing_count = 0
        if existing_count == len(self.docs) and not force:
            return False

        texts = [doc.searchable_text for doc in self.docs]
        logger.info("Indexing Qdrant: embedding %s passages (batch_size=%s)", len(texts), self.settings.embedding_batch_size)
        try:
            vectors = self.embeddings.encode_passages(texts, show_progress_bar=True)
        finally:
            self.embeddings.release_gpu_memory(move_model_to_cpu=True)
        self.qdrant.upsert(self.docs, vectors, show_progress=True)
        return True

    async def ask(self, question: str) -> tuple[str, list[Candidate], TokenUsage]:
        candidates, usage = await self.retrieve_with_usage(question)
        prompt_candidates: list[Candidate] = []
        for candidate in candidates:
            if candidate.rerank_score is not None and candidate.rerank_score < 0:
                continue
            prompt_candidates.append(candidate)
            if len(prompt_candidates) >= self.settings.answer_context_limit:
                break
        answer_context = [
            {
                "source_number": number,
                "chunk_id": candidate.doc.chunk_id,
                "source_ref": candidate.doc.source_ref,
                "text": candidate.doc.searchable_text,
            }
            for number, candidate in enumerate(prompt_candidates, start=1)
        ]
        answer_result = await self.llm.answer(question, answer_context)
        answer = strip_trailing_sources_block(answer_result.content)
        return answer, prompt_candidates, usage + answer_result.usage

    def doc_url_for_chunk(self, doc: ChunkDocument) -> str | None:
        return wiki_url_for_doc_title(doc.doc_title, self._wiki_urls)

    def doc_url_for_doc_title(self, doc_title: str) -> str | None:
        return wiki_url_for_doc_title(doc_title, self._wiki_urls)

    async def retrieve(self, question: str) -> list[Candidate]:
        candidates, _usage = await self.retrieve_with_usage(question)
        return candidates

    async def retrieve_with_usage(self, question: str) -> tuple[list[Candidate], TokenUsage]:
        usage = TokenUsage()
        queries = [question]
        if self.settings.enable_query_rewrite:
            rewrites, rewrite_usage = await self.llm.generate_query_rewrites(
                question,
                self.settings.query_rewrite_count,
            )
            usage += rewrite_usage
            queries.extend(query for query in rewrites if query.casefold() != question.casefold())

        fused: dict[int, Candidate] = {}
        query_vectors = self.embeddings.encode_queries(queries)
        for query_index, (query, vector) in enumerate(zip(queries, query_vectors, strict=True)):
            query_weight = 1.0 if query_index == 0 else 0.7
            dense_hits = self.qdrant.search(vector, limit=self.settings.dense_limit)
            sparse_hits = self.bm25.search(query, limit=self.settings.sparse_limit)

            for rank, hit in enumerate(dense_hits, start=1):
                candidate = fused.setdefault(hit.doc.chunk_id, Candidate(doc=hit.doc))
                candidate.dense_score = max(candidate.dense_score or hit.score, hit.score)
                candidate.fused_score += query_weight / (self.settings.rrf_k + rank)

            for rank, hit in enumerate(sparse_hits, start=1):
                candidate = fused.setdefault(hit.doc.chunk_id, Candidate(doc=hit.doc))
                candidate.sparse_score = max(candidate.sparse_score or hit.score, hit.score)
                candidate.fused_score += (query_weight * 0.9) / (self.settings.rrf_k + rank)

        pool = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)[
            : self.settings.candidate_limit
        ]
        if self.settings.enable_llm_rerank:
            pool, rerank_usage = await self._rerank(question, pool)
            usage += rerank_usage
        return pool, usage

    async def _rerank(self, question: str, candidates: list[Candidate]) -> tuple[list[Candidate], TokenUsage]:
        rerank_input = [
            {
                "chunk_id": candidate.doc.chunk_id,
                "source_ref": candidate.doc.source_ref,
                "text": candidate.doc.searchable_text,
            }
            for candidate in candidates
        ]
        ranking, usage = await self.llm.rerank(question, rerank_input)
        if not ranking:
            return candidates, usage

        score_by_id = {chunk_id: score for chunk_id, score in ranking}
        order_by_id = {chunk_id: index for index, (chunk_id, _) in enumerate(ranking)}
        for candidate in candidates:
            candidate.rerank_score = score_by_id.get(candidate.doc.chunk_id)

        reranked = sorted(
            candidates,
            key=lambda item: (
                item.rerank_score is not None,
                item.rerank_score if item.rerank_score is not None else 0.0,
                -order_by_id.get(item.doc.chunk_id, len(order_by_id)),
                item.fused_score,
            ),
            reverse=True,
        )
        return reranked, usage
