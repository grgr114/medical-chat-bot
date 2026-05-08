"""RAG chunk pipeline: captions → build chunks → expand abbreviations → optional LLM context."""

from pipeline.paths import PIPELINE_ROOT

__all__ = ["PIPELINE_ROOT"]
