from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PipelineContext:
    """Resolved paths for sources (pipeline root = folder containing `images/`, `texts/`, `output/`)."""

    root: Path
    texts_dir: Path
    images_dir: Path
    output_dir: Path
    image_descriptions: Path


@runtime_checkable
class ChunkSource(Protocol):
    """A pluggable data source that emits chunk dicts before chunk_id assignment."""

    source_id: str

    def collect(self, ctx: PipelineContext, options: dict[str, Any]) -> list[dict]:
        """Return rows with keys at least: chunk_text, source_file, chunk_context, doc_title, h1..h6, source."""
        ...
