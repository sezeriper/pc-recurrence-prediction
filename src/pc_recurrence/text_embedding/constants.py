from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("outputs/text_embeddings")
DEFAULT_MODEL_CACHE = Path(".cache/text_models")
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
# Hugging Face commit resolved on 2026-09-09. Never use the mutable `main` revision at runtime.
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MODEL_LABEL = "qwen3-embedding-8b"
EMBEDDING_DIMENSION = 4096
MODEL_CONTEXT_LENGTH = 32768
DEFAULT_MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 1
