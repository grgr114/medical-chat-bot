from __future__ import annotations

import csv
from pathlib import Path


def load_wiki_title_url_map(csv_path: str | Path) -> dict[str, str]:
    path = Path(csv_path)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if title and url:
                out[title] = url
    return out


def wiki_url_for_doc_title(doc_title: str, title_to_url: dict[str, str]) -> str | None:
    t = doc_title.strip()
    if not t:
        return None
    if t in title_to_url:
        return title_to_url[t]
    prefixed = f"WEB {t}"
    if prefixed in title_to_url:
        return title_to_url[prefixed]
    t_cf = t.casefold()
    for key, url in title_to_url.items():
        kcf = key.casefold()
        if kcf == t_cf:
            return url
        if kcf == f"web {t_cf}":
            return url
    return None
