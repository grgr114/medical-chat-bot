from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent


def default_layout() -> dict[str, Path]:
    """Default paths relative to pipeline/ (images and texts live here)."""
    return {
        "images": PIPELINE_ROOT / "images",
        "texts": PIPELINE_ROOT / "texts",
        "output_dir": PIPELINE_ROOT / "output",
        "image_descriptions": PIPELINE_ROOT / "output" / "image_descriptions.jsonl",
        "chunks": PIPELINE_ROOT / "output" / "chunks.jsonl",
        "chunks_expanded": PIPELINE_ROOT / "output" / "chunks_expanded.jsonl",
        "terms": PIPELINE_ROOT / "terms.csv",
        "checkpoints": PIPELINE_ROOT / "output" / "checkpoints",
    }
