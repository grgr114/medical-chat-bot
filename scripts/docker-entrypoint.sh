#!/bin/sh
set -e
python /app/scripts/preload_hf_model.py
exec "$@"
