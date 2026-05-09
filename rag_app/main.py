from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from rag_app.bm25 import BM25Index
from rag_app.config import get_settings
from rag_app.documents import load_chunks
from rag_app.embeddings import EmbeddingModel
from rag_app.history import HistoryStore
from rag_app.llm import LLMClient
from rag_app.qdrant_store import QdrantStore
from rag_app.retrieval import RAGService
from rag_app.citations import (
    append_sources_markdown_block,
    cited_source_indices,
    dedupe_sources_and_remap_citations,
)
from rag_app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SessionSummary,
    Source,
    ResponseStats,
    StoredCall,
)


logger = logging.getLogger(__name__)
static_dir = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()

    docs = load_chunks(settings.chunks_path)
    embeddings = EmbeddingModel(settings)
    qdrant = QdrantStore(settings)
    qdrant.wait_until_ready()
    bm25 = BM25Index(docs)
    llm = LLMClient(settings)
    rag = RAGService(
        settings=settings,
        docs=docs,
        embeddings=embeddings,
        qdrant=qdrant,
        bm25=bm25,
        llm=llm,
    )
    history = HistoryStore(settings)
    await history.connect()

    force = settings.force_reindex_on_start
    if settings.auto_index:
        should_index = True
    else:
        should_index = force 

    if should_index:
        indexed = await run_in_threadpool(rag.build_index, force)
        logger.info(
            "Qdrant index ready: indexed=%s chunks=%s force=%s",
            indexed,
            len(docs),
            force,
        )
    else:
        logger.info("Skipping Qdrant startup indexing.")

    app.state.settings = settings
    app.state.rag = rag
    app.state.history = history
    try:
        yield
    finally:
        await history.close()


app = FastAPI(title="Medical RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    rag: RAGService = app.state.rag
    return HealthResponse(
        status="ok",
        chunks_loaded=len(rag.docs),
        collection=rag.settings.qdrant_collection,
        model=rag.settings.llm_model,
    )


@app.get("/api/config")
async def config() -> dict[str, str | int | bool]:
    settings = app.state.settings
    return {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "collection": settings.qdrant_collection,
        "query_rewrite": settings.enable_query_rewrite,
        "llm_rerank": settings.enable_llm_rerank,
        "chunks_path": settings.chunks_path,
        "texts_csv_path": settings.texts_csv_path,
    }


@app.post("/api/index", response_model=IndexResponse)
async def index_chunks(request: IndexRequest) -> IndexResponse:
    rag: RAGService = app.state.rag
    try:
        indexed = await run_in_threadpool(rag.build_index, request.force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    return IndexResponse(
        collection=rag.settings.qdrant_collection,
        chunks=len(rag.docs),
        indexed=indexed,
        message="Chunks were indexed" if indexed else "Existing index is already up to date",
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    rag: RAGService = app.state.rag
    history: HistoryStore = app.state.history
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message must not be empty")

    started = time.perf_counter()
    session_id = await history.ensure_session(request.session_id, question)
    try:
        answer, candidates, token_usage = await rag.ask(question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG call failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    sources: list[Source] = [
        candidate.to_source(doc_url=rag.doc_url_for_chunk(candidate.doc)) for candidate in candidates
    ]
    answer, sources = dedupe_sources_and_remap_citations(answer, sources)
    cited = cited_source_indices(answer, len(sources))
    answer = append_sources_markdown_block(answer, sources, cited)
    source_payload = [source.model_dump(mode="json") for source in sources]
    stats = ResponseStats(
        latency_ms=latency_ms,
        input_tokens=token_usage.input_tokens,
        output_tokens=token_usage.output_tokens,
    )
    call_id = await history.insert_call(
        session_id=session_id,
        user_message=question,
        answer=answer,
        sources=source_payload,
        model=rag.settings.llm_model,
        latency_ms=latency_ms,
        input_tokens=token_usage.input_tokens,
        output_tokens=token_usage.output_tokens,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        cited_source_indices=cited,
        session_id=session_id,
        call_id=call_id,
        model=rag.settings.llm_model,
        latency_ms=latency_ms,
        stats=stats,
    )


@app.get("/api/sessions", response_model=list[SessionSummary])
async def sessions() -> list[SessionSummary]:
    history: HistoryStore = app.state.history
    rows = await history.list_sessions()
    return [SessionSummary(**row) for row in rows]


@app.get("/api/sessions/{session_id}/calls", response_model=list[StoredCall])
async def session_calls(session_id: UUID) -> list[StoredCall]:
    history: HistoryStore = app.state.history
    rag: RAGService = app.state.rag
    rows = await history.list_calls(session_id)
    calls: list[StoredCall] = []
    for row in rows:
        raw_sources = row["sources"]
        sources_list = raw_sources if isinstance(raw_sources, list) else []
        enriched: list[dict] = []
        for item in sources_list:
            entry = dict(item)
            if not entry.get("doc_url") and entry.get("doc_title"):
                entry["doc_url"] = rag.doc_url_for_doc_title(str(entry["doc_title"]))
            enriched.append(entry)
        calls.append(
            StoredCall(
                id=row["id"],
                session_id=row["session_id"],
                user_message=row["user_message"],
                answer=row["answer"],
                sources=enriched,
                cited_source_indices=cited_source_indices(row["answer"], len(enriched)),
                model=row["model"],
                latency_ms=row["latency_ms"],
                stats=ResponseStats(
                    latency_ms=row["latency_ms"],
                    input_tokens=row.get("input_tokens", 0),
                    output_tokens=row.get("output_tokens", 0),
                ),
                created_at=row["created_at"],
            )
        )
    return calls


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: UUID) -> dict[str, str]:
    history: HistoryStore = app.state.history
    deleted = await history.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}
