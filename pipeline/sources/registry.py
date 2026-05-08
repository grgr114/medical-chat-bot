from __future__ import annotations

import importlib

from pipeline.sources.base import ChunkSource


def import_source_class(dotted: str) -> type[ChunkSource]:
    """Import `package.module:ClassName` and return the class."""
    if ":" not in dotted:
        raise ValueError(f'Chunk source class must be "module:Class", got: {dotted!r}')
    mod_name, _, cls_name = dotted.partition(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ImportError(f"No class {cls_name!r} in {mod_name}")
    return cls  # type: ignore[return-value]


def instantiate(class_path: str) -> ChunkSource:
    cls = import_source_class(class_path)
    return cls()
