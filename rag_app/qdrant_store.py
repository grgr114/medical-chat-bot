from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_app.config import Settings
from rag_app.documents import ChunkDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseHit:
    doc: ChunkDocument
    score: float


class QdrantStore:
    def __init__(self, settings: Settings):
        self.collection = settings.qdrant_collection
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def ensure_collection(self, vector_size: int, force: bool = False) -> None:
        exists = self._collection_exists()
        if exists and force:
            self._client.delete_collection(self.collection)
            exists = False
        elif exists:
            configured = self._configured_vector_size()
            if configured is not None and configured != vector_size:
                logger.warning(
                    "Qdrant collection %r expects vector size %s but embeddings are %s; recreating collection.",
                    self.collection,
                    configured,
                    vector_size,
                )
                self._client.delete_collection(self.collection)
                exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def count(self) -> int:
        result = self._client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    def has_indexed_points(self) -> bool:
        """True if the collection exists and has at least one point (not a fresh/empty volume)."""
        if not self._collection_exists():
            return False
        try:
            return self.count() > 0
        except Exception:
            return False

    def upsert(
        self,
        docs: list[ChunkDocument],
        vectors: list[list[float]],
        batch_size: int = 64,
        *,
        show_progress: bool = False,
    ) -> None:
        n = len(docs)
        total_batches = (n + batch_size - 1) // batch_size if n else 0
        for batch_index, start in enumerate(range(0, n, batch_size), start=1):
            if show_progress and total_batches:
                end = min(start + batch_size, n)
                logger.info(
                    "Qdrant upsert batch %s/%s (points %s–%s of %s)",
                    batch_index,
                    total_batches,
                    start + 1,
                    end,
                    n,
                )
            batch_docs = docs[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                models.PointStruct(id=doc.chunk_id, vector=vector, payload=doc.payload())
                for doc, vector in zip(batch_docs, batch_vectors, strict=True)
            ]
            self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query_vector: list[float], limit: int) -> list[DenseHit]:
        try:
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        except AttributeError:
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            hits = response.points
        return [DenseHit(doc=self._doc_from_payload(hit.payload or {}), score=float(hit.score)) for hit in hits]

    def wait_until_ready(self, attempts: int = 30, delay_seconds: float = 1.0) -> None:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                self._client.get_collections()
                return
            except Exception as exc:  # Qdrant may still be starting under docker-compose.
                last_error = exc
                time.sleep(delay_seconds)
        raise RuntimeError("Qdrant did not become ready") from last_error

    def _collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self.collection))
        except AttributeError:
            collections = self._client.get_collections().collections
            return any(collection.name == self.collection for collection in collections)

    def _configured_vector_size(self) -> int | None:
        """Return dense vector size from collection config, or None if unknown."""
        try:
            info = self._client.get_collection(self.collection)
        except Exception:
            return None
        params = getattr(info.config, "params", None)
        vectors = getattr(params, "vectors", None) if params is not None else None
        if vectors is None:
            return None
        if isinstance(vectors, models.VectorParams):
            return int(vectors.size)
        if isinstance(vectors, dict):
            for spec in vectors.values():
                if isinstance(spec, models.VectorParams):
                    return int(spec.size)
                size = getattr(spec, "size", None)
                if size is not None:
                    return int(size)
            return None
        size = getattr(vectors, "size", None)
        return int(size) if size is not None else None

    @staticmethod
    def _doc_from_payload(payload: dict[str, Any]) -> ChunkDocument:
        return ChunkDocument.from_json(payload, fallback_id=int(payload.get("chunk_id") or 0))

