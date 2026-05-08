from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HEADING_FIELDS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Word / OOXML exports often insert inline image markers like [Изображение «uuid»].
_RE_DOC_IMAGE_PLACEHOLDER = re.compile(
    r"\[[Ии]зображение[^\]]*\]|\[[Ии]зображение\s*«[^»]*»?|\[[Рр]исунок[^\]]*\]",
)


def _strip_doc_image_placeholders(text: str) -> str:
    text = _RE_DOC_IMAGE_PLACEHOLDER.sub("", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class ChunkDocument:
    chunk_id: int
    chunk_text: str
    chunk_context: str
    source_file: str
    doc_title: str
    h1: str = ""
    h2: str = ""
    h3: str = ""
    h4: str = ""
    h5: str = ""
    h6: str = ""
    source: str = ""

    @classmethod
    def from_json(cls, row: dict[str, Any], fallback_id: int) -> "ChunkDocument":
        return cls(
            chunk_id=int(row.get("chunk_id") or fallback_id),
            chunk_text=str(row.get("chunk_text") or "").strip(),
            chunk_context=str(row.get("chunk_context") or "").strip(),
            source_file=str(row.get("source_file") or "").strip(),
            doc_title=str(row.get("doc_title") or "Untitled").strip(),
            h1=str(row.get("h1") or "").strip(),
            h2=str(row.get("h2") or "").strip(),
            h3=str(row.get("h3") or "").strip(),
            h4=str(row.get("h4") or "").strip(),
            h5=str(row.get("h5") or "").strip(),
            h6=str(row.get("h6") or "").strip(),
            source=str(row.get("source") or "").strip(),
        )

    @property
    def source_ref(self) -> str:
        parts = [self.doc_title, self.h1, self.h2]
        return " > ".join(part for part in parts if part)

    @property
    def heading_path(self) -> str:
        parts = [getattr(self, field) for field in HEADING_FIELDS]
        return " > ".join(part for part in parts if part)

    @property
    def searchable_text(self) -> str:
        fields = [
            self.doc_title,
            self.heading_path,
            self.chunk_context,
            self.chunk_text,
        ]
        return "\n\n".join(field for field in fields if field)

    def snippet(self, max_chars: int = 500) -> str:
        text = _strip_doc_image_placeholders(self.chunk_text)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "..."

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "chunk_context": self.chunk_context,
            "source_file": self.source_file,
            "doc_title": self.doc_title,
            "h1": self.h1,
            "h2": self.h2,
            "h3": self.h3,
            "h4": self.h4,
            "h5": self.h5,
            "h6": self.h6,
            "source": self.source,
            "source_ref": self.source_ref,
            "searchable_text": self.searchable_text,
        }


def load_chunks(path: str | Path) -> list[ChunkDocument]:
    chunk_path = Path(path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

    docs: list[ChunkDocument] = []
    with chunk_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {chunk_path}") from exc

            doc = ChunkDocument.from_json(row, fallback_id=line_number)
            if doc.chunk_text:
                docs.append(doc)

    if not docs:
        raise ValueError(f"No usable chunks found in {chunk_path}")
    return docs

