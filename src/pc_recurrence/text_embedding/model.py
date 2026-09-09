from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from pc_recurrence.io import sha256_file
from pc_recurrence.runtime import device_memory_info, inference_autocast, select_device

from .constants import EMBEDDING_DIMENSION, MODEL_ID, MODEL_REVISION


class TextEmbeddingRuntimeError(RuntimeError):
    """Raised when the pinned text embedding model cannot be used safely."""


@dataclass(frozen=True)
class TextModelArtifacts:
    model_directory: Path
    revision: str
    config_sha256: str
    tokenizer_sha256: str
    model_index_sha256: str

    @property
    def fingerprint(self) -> str:
        return ":".join(
            (self.revision, self.config_sha256, self.tokenizer_sha256, self.model_index_sha256)
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "repository": MODEL_ID,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_index_sha256": self.model_index_sha256,
        }


@dataclass(frozen=True)
class TextRuntimeInfo:
    torch_version: str
    transformers_version: str
    device_type: str
    device_name: str
    total_device_bytes: int | None
    free_device_bytes: int | None
    load_seconds: float
    model_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def acquire_qwen_model(
    cache_dir: Path,
    *,
    local_files_only: bool = False,
) -> TextModelArtifacts:
    """Fetch the fixed Qwen snapshot, bypassing stale optional Hugging Face credentials."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Text embeddings require the 'text' extra. For the full project, run "
            "`uv sync --extra imaging --extra text --group dev`."
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_directory = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=False,
            )
        )
    except Exception as error:
        mode = "cached" if local_files_only else "downloadable"
        raise TextEmbeddingRuntimeError(
            f"Unable to acquire the {mode} pinned model {MODEL_ID}@{MODEL_REVISION}: {error}"
        ) from error
    required = {
        "config.json": model_directory / "config.json",
        "tokenizer.json": model_directory / "tokenizer.json",
        "model.safetensors.index.json": model_directory / "model.safetensors.index.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise TextEmbeddingRuntimeError(
            f"Pinned model snapshot is incomplete; missing {', '.join(missing)}"
        )
    return TextModelArtifacts(
        model_directory=model_directory,
        revision=MODEL_REVISION,
        config_sha256=sha256_file(required["config.json"]),
        tokenizer_sha256=sha256_file(required["tokenizer.json"]),
        model_index_sha256=sha256_file(required["model.safetensors.index.json"]),
    )


def load_qwen_runtime(
    artifacts: TextModelArtifacts,
) -> tuple[Any, torch.nn.Module, TextRuntimeInfo]:
    """Load Qwen through the model-card-recommended AutoModel interface."""
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Text embeddings require the 'text' extra. For the full project, run "
            "`uv sync --extra imaging --extra text --group dev`."
        ) from error

    device, device_name = select_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            artifacts.model_directory,
            local_files_only=True,
            padding_side="left",
        )
        model = AutoModel.from_pretrained(
            artifacts.model_directory,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    except Exception as error:
        raise TextEmbeddingRuntimeError(
            f"Unable to load {MODEL_ID}@{artifacts.revision} on {device.type}: {error}"
        ) from error
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size != EMBEDDING_DIMENSION:
        raise TextEmbeddingRuntimeError(
            f"Expected Qwen embedding dimension {EMBEDDING_DIMENSION}; found {hidden_size!r}"
        )
    free_bytes, total_bytes = device_memory_info(device)
    return tokenizer, model, TextRuntimeInfo(
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        device_type=device.type,
        device_name=device_name,
        total_device_bytes=total_bytes,
        free_device_bytes=free_bytes,
        load_seconds=time.perf_counter() - started,
        model_dtype=str(dtype).removeprefix("torch."),
    )


def report_token_details(tokenizer: Any, text: str, max_length: int) -> tuple[int, bool]:
    """Return the untruncated token count and whether the configured limit will truncate it."""
    input_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    token_count = len(input_ids)
    return token_count, token_count > max_length


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = bool(
        attention_mask[:, -1].sum().item() == attention_mask.shape[0]
    )
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def encode_reports(
    tokenizer: Any,
    model: torch.nn.Module,
    texts: list[str],
    *,
    max_length: int,
) -> np.ndarray:
    """Embed report documents with Qwen last-token pooling and L2 normalization."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    if any(not text.strip() for text in texts):
        raise ValueError("CT report texts must be non-empty before embedding")
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    batch = {name: value.to(device) for name, value in batch.items()}
    with torch.inference_mode(), inference_autocast(device):
        outputs = model(**batch)
        embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = functional.normalize(embeddings, p=2, dim=1)
    values = embeddings.to(device="cpu", dtype=torch.float32).numpy()
    if values.shape != (len(texts), EMBEDDING_DIMENSION) or not np.isfinite(values).all():
        raise TextEmbeddingRuntimeError(
            "Unexpected Qwen embedding output: "
            f"shape={values.shape}, finite={np.isfinite(values).all()}"
        )
    return values
