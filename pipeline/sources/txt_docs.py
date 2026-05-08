from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.lib.smart_chunks import build_records_from_txt_dir
from pipeline.sources.base import PipelineContext


class TxtDocsSource:
    """Structured *.txt (doc title line + h1–h6 + {{image}} placeholders) + image_descriptions.jsonl."""

    source_id = "txt_docs"

    def collect(self, ctx: PipelineContext, options: dict[str, Any]) -> list[dict]:
        td = options.get("texts_dir")
        texts_dir = Path(td) if td else ctx.texts_dir
        if not texts_dir.is_absolute():
            texts_dir = ctx.root / texts_dir

        dp = options.get("image_descriptions")
        desc = Path(dp) if dp else ctx.image_descriptions
        if not desc.is_absolute():
            desc = ctx.root / desc

        merge_ctx = options.get("merge_context_path")
        merge_path = Path(merge_ctx) if merge_ctx else None
        if merge_path is not None and not merge_path.is_absolute():
            merge_path = ctx.root / merge_path

        mod_id = options.get("source", self.source_id)
        return build_records_from_txt_dir(
            texts_dir,
            desc,
            merge_context_path=merge_path,
            source_module_id=mod_id,
        )
