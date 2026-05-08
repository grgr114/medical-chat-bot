"""Download/refresh the embedding model in HF_HOME before the app starts (Docker entrypoint)."""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    model_id = (os.environ.get("EMBEDDING_MODEL") or "jinaai/jina-embeddings-v3").strip()
    if not model_id:
        return
    from huggingface_hub import snapshot_download

    log.info("Ensuring Hugging Face model is available: %s", model_id)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        repo_id=model_id,
        token=token,
    )
    log.info("Model files ready: %s", model_id)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Hugging Face preload failed")
        sys.exit(1)
