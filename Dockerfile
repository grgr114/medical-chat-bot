# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl

ARG INSTALL_CUDA_TORCH=0

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && if [ "$INSTALL_CUDA_TORCH" = "1" ]; then \
      python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch; \
    else \
      python -m pip install --index-url https://download.pytorch.org/whl/cpu torch; \
    fi \
    && python -m pip install -r requirements.txt

COPY scripts ./scripts
COPY rag_app ./rag_app
COPY pipeline/output/chunks.jsonl ./pipeline/output/chunks.jsonl
COPY pipeline/questions.csv ./pipeline/questions.csv
COPY pipeline/texts.csv ./pipeline/texts.csv

# Windows checkouts often use CRLF; strip CR so the entrypoint runs in Linux.
RUN sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh \
    && chmod +x /app/scripts/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
