from __future__ import annotations

import re

_BRACKET_NUM = re.compile(r"\[\s*(\d+)\s*\]")
# Collapse runs of the same citation [n] separated by commas, spaces, or «и».
_COLLAPSE_SAME_ADJACENT_CITES = re.compile(
    r"(\[\s*(\d+)\s*\])(?:(?:\s*(?:,\s*|и\s+|\s+))?\[\s*\2\s*\])+"
)
_SOURCES_TAIL = re.compile(
    r"(?:\r?\n){2,}(?:[*_#>\s-]*)(?:Источники|источники)\s*:?\s*[\s\S]*$",
    re.IGNORECASE,
)
_SOURCES_TAIL_SINGLE = re.compile(
    r"(?:\r?\n)(?:[*_#>\s-]*)(?:Источники|источники)\s*:?\s*[\s\S]*$",
    re.IGNORECASE,
)


def strip_trailing_sources_block(text: str) -> str:
    updated = _SOURCES_TAIL.sub("", text)
    if updated == text:
        updated = _SOURCES_TAIL_SINGLE.sub("", text)
    return updated.rstrip()


def cited_source_indices(answer: str, num_sources: int) -> list[int]:
    if num_sources <= 0:
        return []
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _BRACKET_NUM.finditer(answer):
        n = int(match.group(1))
        if n < 1 or n > num_sources or n in seen:
            continue
        seen.add(n)
        ordered.append(n)
    return ordered


def _source_display_key(src: object) -> tuple[str, str]:
    """Match `append_sources_markdown`: label + URL identity for deduplication."""
    label = (getattr(src, "source_ref", None) or getattr(src, "doc_title", None) or "").strip()
    url = getattr(src, "doc_url", None)
    url_s = str(url).strip() if url else ""
    return (label, url_s)


def collapse_adjacent_duplicate_citations(text: str) -> str:
    """Turn `[1], [1], [1]` / `[1][1]` into `[1]` after numbers are already normalized."""
    prev = None
    current = text
    while prev != current:
        prev = current
        current = _COLLAPSE_SAME_ADJACENT_CITES.sub(r"\1", current)
    return current


def dedupe_sources_and_remap_citations(answer: str, sources: list[object]) -> tuple[str, list[object]]:
    """
    Merge retrieval chunks that would render as the same «Источники» line (same label + URL).
    Remaps every [n] in the answer to the canonical index; collapses repeated adjacent cites.
    """
    if not sources:
        return answer, []

    n_sources = len(sources)
    appearance = cited_source_indices(answer, n_sources)
    if not appearance:
        appearance = list(range(1, n_sources + 1))

    key_to_new: dict[tuple[str, str], int] = {}
    deduped: list[object] = []
    next_id = 1

    def assign_from_index(idx: int) -> None:
        nonlocal next_id
        key = _source_display_key(sources[idx - 1])
        if key not in key_to_new:
            key_to_new[key] = next_id
            deduped.append(sources[idx - 1])
            next_id += 1

    for idx in appearance:
        assign_from_index(idx)
    for idx in range(1, n_sources + 1):
        key = _source_display_key(sources[idx - 1])
        if key not in key_to_new:
            assign_from_index(idx)

    old_to_new: dict[int, int] = {}
    for idx in range(1, n_sources + 1):
        key = _source_display_key(sources[idx - 1])
        old_to_new[idx] = key_to_new[key]

    def _remap_bracket(m: re.Match[str]) -> str:
        n = int(m.group(1))
        mapped = old_to_new.get(n, n)
        return f"[{mapped}]"

    remapped = _BRACKET_NUM.sub(_remap_bracket, answer)
    remapped = collapse_adjacent_duplicate_citations(remapped)
    return remapped, deduped


def _escape_md_link_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def append_sources_markdown_block(answer: str, sources: list[object], cited: list[int]) -> str:
    """Append a Russian «Источники» list for the chat bubble (markdown)."""
    if not sources:
        return answer
    indices = cited if cited else list(range(1, len(sources) + 1))
    lines: list[str] = ["", "**Источники:**", ""]
    for n in indices:
        if n < 1 or n > len(sources):
            continue
        src = sources[n - 1]
        label = (getattr(src, "source_ref", None) or getattr(src, "doc_title", None) or "").strip()
        if not label:
            label = f"Источник {n}"
        url = getattr(src, "doc_url", None)
        url = str(url).strip() if url else ""
        if url:
            lines.append(f"- **[{n}]** [{_escape_md_link_label(label)}]({url})")
        else:
            lines.append(f"- **[{n}]** {label}")
    if len(lines) <= 3:
        return answer
    return answer.rstrip() + "\n" + "\n".join(lines)
