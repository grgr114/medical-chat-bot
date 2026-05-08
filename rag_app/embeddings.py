from __future__ import annotations

import gc
import json
import logging
import os
import warnings
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, settings: Settings):
        kwargs = {"trust_remote_code": True}
        if settings.embedding_device:
            kwargs["device"] = settings.embedding_device
        # Some HF models (e.g. remote code) trigger cuda autocast / sdp_kernel warnings on CPU.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*CUDA is not available.*autocast.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*torch\\.backends\\.cuda\\.sdp_kernel.*",
                category=FutureWarning,
            )
            try:
                self._model = SentenceTransformer(settings.embedding_model, **kwargs)
            except json.JSONDecodeError as exc:
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                raise RuntimeError(
                    "Embedding model files look corrupted or incomplete (empty/invalid JSON in the "
                    "Hugging Face cache). Delete the cache and restart so files download again. "
                    f"HF_HOME is {hf_home!r}. With Docker Compose, remove the hf_cache volume "
                    "(`docker compose down -v` or `docker volume rm <project>_hf_cache`) and bring "
                    "the stack back up."
                ) from exc
        self._batch_size = settings.embedding_batch_size
        self._query_prefix = settings.embedding_query_prefix
        self._passage_prefix = settings.embedding_passage_prefix
        self._query_task = (settings.embedding_query_task or "").strip()
        self._passage_task = (settings.embedding_passage_task or "").strip()
        self._dimension: int | None = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model %s on device %s", settings.embedding_model, getattr(self._model, "device", "?"))

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode_queries(["dimension probe"])[0])
        return self._dimension

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [self._query_prefix + text for text in texts]
        return self._encode(prepared, task=self._query_task or None)

    def encode_passages(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> list[list[float]]:
        prepared = [self._passage_prefix + text for text in texts]
        task = self._passage_task or None
        if not show_progress_bar:
            try:
                return self._encode(prepared, show_progress_bar=False, task=task)
            finally:
                self.release_gpu_memory()

        n = len(prepared)
        bs = self._batch_size
        if n == 0:
            return []
        total_batches = (n + bs - 1) // bs
        out: list[list[float]] = []
        for batch_index, start in enumerate(range(0, n, bs), start=1):
            end = min(start + bs, n)
            logger.info(
                "Embedding passages batch %s/%s (texts %s–%s of %s)",
                batch_index,
                total_batches,
                start + 1,
                end,
                n,
            )
            chunk = prepared[start : start + bs]
            try:
                out.extend(self._encode(chunk, show_progress_bar=False, task=task))
            finally:
                self.release_gpu_memory()
        return out

    def _encode(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
        task: str | None = None,
    ) -> list[list[float]]:
        encode_kwargs: dict = {
            "batch_size": self._batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress_bar,
        }
        if task:
            encode_kwargs["task"] = task
        vectors = self._model.encode(list(texts), **encode_kwargs)
        if isinstance(vectors, np.ndarray):
            return vectors.astype(float).tolist()
        return [np.asarray(vector, dtype=float).tolist() for vector in vectors]

    def release_gpu_memory(self, *, move_model_to_cpu: bool = False) -> None:
        try:
            import torch
        except ImportError:
            return

        device = str(getattr(self._model, "device", "") or "")
        if move_model_to_cpu and device.startswith(("cuda", "mps")):
            logger.info("Moving embedding model from %s to CPU to release GPU memory", device)
            self._model.to("cpu")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_mps = getattr(torch, "mps", None)
            if torch_mps is not None:
                torch_mps.empty_cache()
