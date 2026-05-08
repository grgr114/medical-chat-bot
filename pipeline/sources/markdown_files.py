from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pipeline.sources.base import PipelineContext

# Optional first Markdown H1 for doc_title
_H1 = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


class MarkdownFilesSource:
    """
    Example source: one or more .md files → one chunk per file (body = full text after optional title line).

    Configure in pipeline/config.json under sources[].options:
      - "glob": "notes/**/*.md"  (relative to pipeline root)
    """

    source_id = "markdown_files"

    def collect(self, ctx: PipelineContext, options: dict[str, Any]) -> list[dict]:
        pattern = options.get("glob") or "markdown_import/**/*.md"
        root = ctx.root
        paths = sorted(root.glob(pattern))
        records: list[dict] = []
        for path in paths:
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8")
            m = _H1.search(raw)
            if m:
                doc_title = m.group(1).strip()
                body = raw[m.end() :].lstrip()
            else:
                doc_title = path.stem
                body = raw.strip()
            rel = path.relative_to(root).as_posix()
            records.append(
                {
                    "chunk_text": body,
                    "source_file": rel,
                    "chunk_context": "",
                    "doc_title": doc_title,
                    "h1": "",
                    "h2": "",
                    "h3": "",
                    "h4": "",
                    "h5": "",
                    "h6": "",
                    "source": options.get("source", self.source_id),
                }
            )
        return records
