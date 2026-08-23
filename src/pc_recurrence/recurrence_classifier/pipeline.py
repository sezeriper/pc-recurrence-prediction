from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pc_recurrence import __version__
from pc_recurrence.image_data.workbook import load_image_workbook
from pc_recurrence.image_embedding.constants import (
    MERLIN_EMBEDDING_DIMENSION,
    SPECTRE_EMBEDDING_DIMENSION,
    ImageEncoderName,
)
from pc_recurrence.io import (
    create_run_directory,
    read_json,
    sha256_file,
    write_json,
    write_npz,
    write_summary,
)

from .constants import DECISION_THRESHOLD, TEST_FRACTION
from .model import RecurrenceHead

EMBEDDING_KEYS = {"patient_ids", "embeddings", "patch_counts", "encoder"}
MODEL_KEYS = {
    "weight",
    "bias",
    "mean",
    "scale",
    "encoder",
    "threshold",
    "embedding_dimension",
    "training_patient_ids",
}
DIMENSIONS = {
    ImageEncoderName.MERLIN.value: MERLIN_EMBEDDING_DIMENSION,
    ImageEncoderName.SPECTRE.value: SPECTRE_EMBEDDING_DIMENSION,
}
EVALUATION_COLUMNS = ("encoder", "patient_id", "target", "logit", "probability", "prediction")
COHORT_COLUMNS = EVALUATION_COLUMNS + ("fit_scope",)
PREDICTION_COLUMNS = ("encoder", "patient_id", "logit", "probability", "prediction")

FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int32]
StringVector = NDArray[np.str_]


@dataclass(frozen=True)
class EmbeddingRun:
    encoder: str
    run: Path
    patient_ids: StringVector
    embeddings: FloatMatrix
    patch_counts: IntVector
    manifest: dict[str, Any]
    records: dict[str, dict[str, Any]]
    embedding_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class FittedModel:
    encoder: str
    weight: NDArray[np.float32]
    bias: np.float32
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    threshold: np.float32
    training_patient_ids: StringVector


def parse_recurrence_label(value: Any, *, patient_id: str) -> int:
    """Map workbook recurrence values to binary targets."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Patient {patient_id!r} has a blank recurrence label")
    if isinstance(value, str) and value.strip().casefold() == "yok":
        return 0
    return 1


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _scalar(array: np.ndarray, name: str) -> Any:
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar; found shape {array.shape}")
    return array.item()


def _load_embedding_run(run: Path, expected_encoder: str) -> EmbeddingRun:
    artifact = run / "image_embeddings.npz"
    manifest_path = run / "run_manifest.json"
    _require_file(artifact, f"{expected_encoder} patient embeddings")
    _require_file(manifest_path, f"{expected_encoder} run manifest")

    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != EMBEDDING_KEYS:
                raise ValueError(
                    f"{expected_encoder} embedding keys must be exactly {sorted(EMBEDDING_KEYS)}; "
                    f"found {sorted(archive.files)}"
                )
            patient_ids = archive["patient_ids"]
            embeddings = archive["embeddings"]
            patch_counts = archive["patch_counts"]
            encoder_array = archive["encoder"]
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid {expected_encoder} embedding artifact {artifact}: {error}"
        ) from error

    encoder = _scalar(encoder_array, f"{expected_encoder} encoder")
    if not isinstance(encoder, str) or encoder != expected_encoder:
        raise ValueError(f"expected encoder {expected_encoder!r}; found {encoder!r}")
    if patient_ids.ndim != 1 or patient_ids.size == 0 or patient_ids.dtype.kind != "U":
        raise ValueError(f"{expected_encoder} patient_ids must be a non-empty 1D Unicode array")
    id_values = patient_ids.tolist()
    if any(not value for value in id_values) or len(set(id_values)) != len(id_values):
        raise ValueError(f"{expected_encoder} patient_ids must be non-empty and unique")
    dimension = DIMENSIONS[expected_encoder]
    if embeddings.dtype != np.float32 or embeddings.shape != (patient_ids.size, dimension):
        raise ValueError(
            f"{expected_encoder} embeddings must be float32 with shape "
            f"({patient_ids.size}, {dimension}); found {embeddings.dtype} {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{expected_encoder} embeddings contain non-finite values")
    if patch_counts.dtype != np.int32 or patch_counts.shape != (patient_ids.size,):
        raise ValueError(
            f"{expected_encoder} patch_counts must be int32 with shape ({patient_ids.size},)"
        )
    if np.any(patch_counts < 0):
        raise ValueError(f"{expected_encoder} patch_counts contain negative values")

    manifest = read_json(manifest_path)
    expected_stage = f"{expected_encoder}_selected_ct_embedding"
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"{expected_encoder} manifest stage must be {expected_stage!r}")
    if manifest.get("status") not in {"complete", "completed_with_failures"}:
        raise ValueError(f"{expected_encoder} manifest is not complete")
    checks = {
        "encoder": expected_encoder,
        "patient_count": patient_ids.size,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{expected_encoder} manifest {key} must be {expected!r}; "
                f"found {manifest.get(key)!r}"
            )
    feature_dimension = manifest.get("feature_extraction", {}).get("embedding_dimension")
    if feature_dimension != dimension:
        raise ValueError(
            f"{expected_encoder} manifest embedding dimension must be {dimension}; "
            f"found {feature_dimension!r}"
        )
    if manifest.get("artifacts", {}).get("patient_embeddings") != "image_embeddings.npz":
        raise ValueError(f"{expected_encoder} manifest patient embedding artifact is invalid")
    patients = manifest.get("patients")
    if not isinstance(patients, list) or any(not isinstance(item, dict) for item in patients):
        raise ValueError(f"{expected_encoder} manifest patients must be a list of records")
    embedded = [item for item in patients if item.get("status") == "embedded"]
    embedded_ids = [item.get("patient_id") for item in embedded]
    if embedded_ids != id_values:
        raise ValueError(
            f"{expected_encoder} manifest embedded patient order does not match "
            "image_embeddings.npz"
        )
    records = {str(item["patient_id"]): item for item in embedded}
    return EmbeddingRun(
        encoder=expected_encoder,
        run=run,
        patient_ids=patient_ids.astype(np.str_, copy=False),
        embeddings=embeddings,
        patch_counts=patch_counts,
        manifest=manifest,
        records=records,
        embedding_sha256=sha256_file(artifact),
        manifest_sha256=sha256_file(manifest_path),
    )


def _centering(record: dict[str, Any]) -> Any:
    preprocessing = record.get("preprocessing")
    return preprocessing.get("centering") if isinstance(preprocessing, dict) else None


def _validate_provenance(
    merlin: EmbeddingRun, spectre: EmbeddingRun, patient_ids: list[str]
) -> None:
    source_hash_field = "source_curation_manifest_sha256"
    if merlin.manifest.get(source_hash_field) != spectre.manifest.get(source_hash_field):
        raise ValueError("Merlin and SPECTRE source curation manifest SHA-256 values differ")
    for patient_id in patient_ids:
        merlin_record = merlin.records[patient_id]
        spectre_record = spectre.records[patient_id]
        if merlin_record.get("ct_series_sha256") != spectre_record.get("ct_series_sha256"):
            raise ValueError(
                f"Patient {patient_id!r} has mismatched ct_series_sha256 across encoders"
            )
        if _centering(merlin_record) != _centering(spectre_record):
            raise ValueError(f"Patient {patient_id!r} has mismatched preprocessing centering")


def _canonical_threshold(threshold: float) -> np.float32:
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and strictly between 0 and 1")
    return np.float32(threshold)


def _choose_split(
    patient_ids: list[str],
    targets: NDArray[np.int64],
    test_fraction: float,
    seed_start: int,
) -> tuple[dict[str, Any], NDArray[np.int64], NDArray[np.int64]]:
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be finite and strictly between 0 and 1")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    count = len(patient_ids)
    positives = int(targets.sum())
    negatives = count - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "common cohort must contain both recurrence classes; "
            f"found {positives} positive and {negatives} negative"
        )
    holdout_count = min(count - 1, max(1, math.ceil(test_fraction * count)))
    train_count = count - holdout_count
    if train_count < 2:
        raise ValueError(
            f"training split has {train_count} slot(s); at least 2 are required "
            "for both recurrence classes"
        )
    require_test_classes = min(positives, negatives) >= 2 and holdout_count >= 2
    seed_end = seed_start + 9999
    for seed in range(seed_start, seed_end + 1):
        permutation = np.random.default_rng(seed).permutation(count)
        test_indices = permutation[:holdout_count]
        train_indices = permutation[holdout_count:]
        train_targets = targets[train_indices]
        test_targets = targets[test_indices]
        if np.unique(train_targets).size != 2:
            continue
        if require_test_classes and np.unique(test_targets).size != 2:
            continue
        split = {
            "test_fraction": test_fraction,
            "seed_search_start": seed_start,
            "seed_search_end": seed_end,
            "seed": seed,
            "train_patient_ids": [patient_ids[index] for index in train_indices],
            "test_patient_ids": [patient_ids[index] for index in test_indices],
            "class_counts": {
                "overall": {"positive": positives, "negative": negatives},
                "train": {
                    "positive": int(train_targets.sum()),
                    "negative": int(train_targets.size - train_targets.sum()),
                },
                "test": {
                    "positive": int(test_targets.sum()),
                    "negative": int(test_targets.size - test_targets.sum()),
                },
            },
        }
        return split, train_indices, test_indices
    raise ValueError(
        f"no qualifying split seed found in [{seed_start}, {seed_end}] for {positives} positive, "
        f"{negatives} negative, {train_count} train, and {holdout_count} test patients"
    )


def _fit_model(
    encoder: str,
    embeddings: FloatMatrix,
    targets: NDArray[np.int64],
    patient_ids: list[str],
    threshold: np.float32,
) -> FittedModel:
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{encoder} training split must contain both recurrence classes; "
            f"found {positives} positive and {negatives} negative"
        )
    mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = embeddings.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < np.float32(1e-6)] = np.float32(1.0)
    standardized = ((embeddings - mean) / scale).astype(np.float32)
    inputs = torch.from_numpy(standardized)
    target_tensor = torch.from_numpy(targets.astype(np.float32))
    head = RecurrenceHead(embeddings.shape[1]).cpu()
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.zero_()
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32)
    )
    optimizer = torch.optim.LBFGS(
        head.parameters(),
        lr=1.0,
        max_iter=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = head(inputs)
        loss = criterion(logits, target_tensor) + 1e-4 * head.linear.weight.square().sum()
        if not torch.isfinite(loss):
            raise ValueError(f"{encoder} optimization produced a non-finite loss")
        loss.backward()
        return loss

    optimizer.step(closure)
    weight = head.linear.weight.detach().numpy().reshape(-1).astype(np.float32)
    bias = np.float32(head.linear.bias.detach().item())
    if not np.isfinite(weight).all() or not np.isfinite(bias):
        raise ValueError(f"{encoder} optimization produced non-finite parameters")
    return FittedModel(
        encoder=encoder,
        weight=weight,
        bias=bias,
        mean=mean,
        scale=scale,
        threshold=threshold,
        training_patient_ids=np.asarray(patient_ids, dtype=np.str_),
    )


def _infer(
    model: FittedModel, embeddings: FloatMatrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardized = (embeddings - model.mean) / model.scale
    logits = standardized @ model.weight + model.bias
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    predictions = (probabilities >= float(model.threshold)).astype(np.int64)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError(f"{model.encoder} inference produced non-finite outputs")
    return logits, probabilities, predictions


def _metrics(
    targets: NDArray[np.int64], probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    result: dict[str, Any] = {
        "patient_count": int(targets.size),
        "positive_count": int(targets.sum()),
        "negative_count": int(targets.size - targets.sum()),
        "bce": float(np.mean(-(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "accuracy": float(np.mean(predictions == targets)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    if tp + fn:
        result["sensitivity"] = tp / (tp + fn)
    else:
        result["sensitivity"] = None
        result["sensitivity_reason"] = "holdout contains no positive targets"
    if tn + fp:
        result["specificity"] = tn / (tn + fp)
    else:
        result["specificity"] = None
        result["specificity_reason"] = "holdout contains no negative targets"
    if tp + fp:
        result["precision"] = tp / (tp + fp)
    else:
        result["precision"] = None
        result["precision_reason"] = "holdout contains no predicted positives"
    if 2 * tp + fp + fn:
        result["f1"] = 2 * tp / (2 * tp + fp + fn)
    else:
        result["f1"] = None
        result["f1_reason"] = "holdout contains no positive targets and no positive predictions"
    positive_probabilities = probabilities[targets == 1]
    negative_probabilities = probabilities[targets == 0]
    if positive_probabilities.size and negative_probabilities.size:
        comparisons = positive_probabilities[:, None] - negative_probabilities[None, :]
        result["roc_auc"] = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))
    else:
        result["roc_auc"] = None
        result["roc_auc_reason"] = "holdout contains fewer than two target classes"
    return result


def _rows(
    encoder: str,
    patient_ids: list[str],
    targets: NDArray[np.int64] | None,
    logits: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    fit_scope: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, patient_id in enumerate(patient_ids):
        row: dict[str, Any] = {
            "encoder": encoder,
            "patient_id": patient_id,
            "logit": float(logits[index]),
            "probability": float(probabilities[index]),
            "prediction": int(predictions[index]),
        }
        if targets is not None:
            row["target"] = int(targets[index])
        if fit_scope is not None:
            row["fit_scope"] = fit_scope
        rows.append(row)
    return rows


def _destination(output_root: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        return run_dir
    return create_run_directory(output_root)


def _model_arrays(model: FittedModel) -> dict[str, np.ndarray]:
    return {
        "weight": model.weight,
        "bias": np.asarray(model.bias, dtype=np.float32),
        "mean": model.mean,
        "scale": model.scale,
        "encoder": np.asarray(model.encoder),
        "threshold": np.asarray(model.threshold, dtype=np.float32),
        "embedding_dimension": np.asarray(model.weight.size, dtype=np.int32),
        "training_patient_ids": model.training_patient_ids,
    }


def train_recurrence_models(
    merlin_run: Path,
    spectre_run: Path,
    workbook_path: Path,
    output_root: Path,
    *,
    run_dir: Path | None = None,
    test_fraction: float = TEST_FRACTION,
    seed_start: int = 0,
    threshold: float = DECISION_THRESHOLD,
) -> Path:
    if run_dir is not None and run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    canonical_threshold = _canonical_threshold(threshold)
    workbook_path = Path(workbook_path)
    _require_file(workbook_path, "workbook")
    merlin = _load_embedding_run(Path(merlin_run), ImageEncoderName.MERLIN.value)
    spectre = _load_embedding_run(Path(spectre_run), ImageEncoderName.SPECTRE.value)
    workbook_rows = load_image_workbook(workbook_path)
    workbook_by_id = {row.patient_id: row for row in workbook_rows}
    for source in (merlin, spectre):
        unknown = [
            patient_id
            for patient_id in source.patient_ids.tolist()
            if patient_id not in workbook_by_id
        ]
        if unknown:
            raise ValueError(
                f"{source.encoder} embeddings contain patient IDs absent from workbook: "
                f"{', '.join(unknown)}"
            )
    merlin_ids = set(merlin.patient_ids.tolist())
    spectre_ids = set(spectre.patient_ids.tolist())
    common_ids = [
        row.patient_id for row in workbook_rows if row.patient_id in merlin_ids & spectre_ids
    ]
    if not common_ids:
        raise ValueError("common embedding cohort is empty")
    _validate_provenance(merlin, spectre, common_ids)
    labels = {
        row.patient_id: parse_recurrence_label(row.recurrence_raw, patient_id=row.patient_id)
        for row in workbook_rows
        if row.patient_id in merlin_ids | spectre_ids
    }
    targets = np.asarray([labels[patient_id] for patient_id in common_ids], dtype=np.int64)
    split, train_indices, test_indices = _choose_split(
        common_ids, targets, test_fraction, seed_start
    )
    source_indices: dict[str, NDArray[np.int64]] = {}
    for source in (merlin, spectre):
        source_index = {
            patient_id: index
            for index, patient_id in enumerate(source.patient_ids.tolist())
        }
        source_indices[source.encoder] = np.asarray(
            [source_index[patient_id] for patient_id in common_ids], dtype=np.int64
        )

    metrics: dict[str, Any] = {}
    evaluation_rows: list[dict[str, Any]] = []
    final_models: dict[str, FittedModel] = {}
    cohort_rows: list[dict[str, Any]] = []
    for source in (merlin, spectre):
        common_embeddings = source.embeddings[source_indices[source.encoder]]
        split_model = _fit_model(
            source.encoder,
            common_embeddings[train_indices],
            targets[train_indices],
            split["train_patient_ids"],
            canonical_threshold,
        )
        logits, probabilities, predictions = _infer(split_model, common_embeddings[test_indices])
        metrics[source.encoder] = _metrics(targets[test_indices], probabilities, predictions)
        evaluation_rows.extend(
            _rows(
                source.encoder,
                split["test_patient_ids"],
                targets[test_indices],
                logits,
                probabilities,
                predictions,
            )
        )
        final_model = _fit_model(
            source.encoder, common_embeddings, targets, common_ids, canonical_threshold
        )
        final_models[source.encoder] = final_model
        full_logits, full_probabilities, full_predictions = _infer(final_model, common_embeddings)
        cohort_rows.extend(
            _rows(
                source.encoder,
                common_ids,
                targets,
                full_logits,
                full_probabilities,
                full_predictions,
                fit_scope="full_cohort",
            )
        )

    exclusions: list[dict[str, str]] = []
    for row in workbook_rows:
        reasons = []
        if row.patient_id not in merlin_ids:
            reasons.append("missing_merlin_embedding")
        if row.patient_id not in spectre_ids:
            reasons.append("missing_spectre_embedding")
        for reason in reasons:
            exclusions.append({"patient_id": row.patient_id, "reason": reason})

    destination = _destination(Path(output_root), run_dir)
    model_entries: dict[str, Any] = {}
    for encoder in (ImageEncoderName.MERLIN.value, ImageEncoderName.SPECTRE.value):
        relative = Path("models") / f"{encoder}.npz"
        model_path = destination / relative
        write_npz(model_path, **_model_arrays(final_models[encoder]))
        model_entries[encoder] = {
            "artifact": relative.as_posix(),
            "sha256": sha256_file(model_path),
            "encoder": encoder,
            "embedding_dimension": DIMENSIONS[encoder],
            "threshold": float(canonical_threshold),
            "fit_scope": "full_cohort",
            "training_patient_ids": common_ids,
        }
    write_summary(evaluation_rows, destination / "evaluation_predictions.csv", EVALUATION_COLUMNS)
    write_summary(cohort_rows, destination / "cohort_classifications.csv", COHORT_COLUMNS)
    metrics_payload = {
        "evaluation_kind": "single_holdout",
        "threshold": float(canonical_threshold),
        "split": split,
        "encoders": metrics,
    }
    write_json(metrics_payload, destination / "metrics.json")
    sources = {
        source.encoder: {
            "run": str(source.run.resolve()),
            "image_embeddings_sha256": source.embedding_sha256,
            "run_manifest_sha256": source.manifest_sha256,
        }
        for source in (merlin, spectre)
    }
    write_json(
        {
            "pipeline_version": __version__,
            "stage": "recurrence_classification_training",
            "status": "complete",
            "sources": sources,
            "workbook": str(workbook_path.resolve()),
            "workbook_sha256": sha256_file(workbook_path),
            "label_policy": {
                "negative": "trimmed case-folded 'yok'",
                "positive": "every other non-empty value",
            },
            "architecture": "independent affine heads over pooled embeddings",
            "training": {
                "optimizer": "LBFGS",
                "max_iter": 100,
                "positive_class_weight": "training_negative_count/training_positive_count",
                "l2_weight_penalty": 1e-4,
                "standardization_scale_floor": 1e-6,
            },
            "threshold": float(canonical_threshold),
            "fit_scope": "full_cohort",
            "cohort": {
                "patient_ids": common_ids,
                "patient_count": len(common_ids),
                "positive_count": int(targets.sum()),
                "negative_count": int(targets.size - targets.sum()),
                "excluded": exclusions,
            },
            "split": split,
            "artifacts": {
                "metrics": "metrics.json",
                "evaluation_predictions": "evaluation_predictions.csv",
                "cohort_classifications": "cohort_classifications.csv",
            },
            "models": model_entries,
            "provisional_research_output": True,
        },
        destination / "run_manifest.json",
    )
    return destination


def _load_training_model(
    model_run: Path, manifest: dict[str, Any], encoder: str, threshold: np.float32
) -> tuple[FittedModel, str]:
    entry = manifest.get("models", {}).get(encoder)
    if not isinstance(entry, dict):
        raise ValueError(f"training manifest has no {encoder} model entry")
    expected_artifact = f"models/{encoder}.npz"
    if entry.get("artifact") != expected_artifact:
        raise ValueError(f"{encoder} model artifact must be {expected_artifact!r}")
    if entry.get("encoder") != encoder or entry.get("embedding_dimension") != DIMENSIONS[encoder]:
        raise ValueError(f"{encoder} model manifest identity or dimension is invalid")
    if entry.get("threshold") != float(threshold):
        raise ValueError(f"{encoder} model manifest threshold does not match training threshold")
    if entry.get("fit_scope") != "full_cohort":
        raise ValueError(f"{encoder} model fit_scope must be 'full_cohort'")
    training_ids = entry.get("training_patient_ids")
    cohort_ids = manifest.get("cohort", {}).get("patient_ids")
    if not isinstance(training_ids, list) or training_ids != cohort_ids:
        raise ValueError(f"{encoder} model training IDs do not match the full cohort")
    model_path = model_run / expected_artifact
    _require_file(model_path, f"{encoder} model")
    digest = sha256_file(model_path)
    if entry.get("sha256") != digest:
        raise ValueError(f"{encoder} model SHA-256 does not match training manifest")
    try:
        with np.load(model_path, allow_pickle=False) as archive:
            if set(archive.files) != MODEL_KEYS:
                raise ValueError(
                    f"model keys must be exactly {sorted(MODEL_KEYS)}; "
                    f"found {sorted(archive.files)}"
                )
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {encoder} model artifact {model_path}: {error}") from error
    dimension = DIMENSIONS[encoder]
    for name in ("weight", "mean", "scale"):
        array = arrays[name]
        if array.dtype != np.float32 or array.shape != (dimension,) or not np.isfinite(array).all():
            raise ValueError(f"{encoder} model {name} must be finite float32 shape ({dimension},)")
    if np.any(arrays["scale"] <= 0):
        raise ValueError(f"{encoder} model scale must be strictly positive")
    bias = arrays["bias"]
    model_threshold = arrays["threshold"]
    model_dimension = arrays["embedding_dimension"]
    model_encoder = arrays["encoder"]
    ids = arrays["training_patient_ids"]
    if bias.dtype != np.float32 or bias.shape != () or not np.isfinite(bias):
        raise ValueError(f"{encoder} model bias must be a finite scalar float32")
    if model_threshold.dtype != np.float32 or model_threshold.shape != ():
        raise ValueError(f"{encoder} model threshold must be scalar float32")
    if (
        model_dimension.dtype != np.int32
        or model_dimension.shape != ()
        or model_dimension.item() != dimension
    ):
        raise ValueError(f"{encoder} model embedding_dimension must be scalar int32 {dimension}")
    if (
        model_encoder.shape != ()
        or model_encoder.dtype.kind != "U"
        or model_encoder.item() != encoder
    ):
        raise ValueError(f"{encoder} model encoder must be scalar Unicode {encoder!r}")
    if ids.ndim != 1 or ids.dtype.kind != "U" or ids.tolist() != training_ids:
        raise ValueError(f"{encoder} model training IDs do not match training manifest")
    if model_threshold.item() != threshold.item():
        raise ValueError(f"{encoder} model threshold does not match training manifest")
    return (
        FittedModel(
            encoder=encoder,
            weight=arrays["weight"],
            bias=np.float32(bias.item()),
            mean=arrays["mean"],
            scale=arrays["scale"],
            threshold=np.float32(model_threshold.item()),
            training_patient_ids=ids,
        ),
        digest,
    )


def predict_recurrence(
    model_run: Path,
    merlin_run: Path,
    spectre_run: Path,
    output_root: Path,
    *,
    run_dir: Path | None = None,
) -> Path:
    if run_dir is not None and run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    model_run = Path(model_run)
    manifest_path = model_run / "run_manifest.json"
    _require_file(manifest_path, "training run manifest")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != "recurrence_classification_training":
        raise ValueError("model run manifest stage must be 'recurrence_classification_training'")
    if manifest.get("status") != "complete":
        raise ValueError("model run manifest status must be 'complete'")
    if manifest.get("fit_scope") != "full_cohort":
        raise ValueError("model run manifest fit_scope must be 'full_cohort'")
    manifest_threshold = manifest.get("threshold")
    if not isinstance(manifest_threshold, (int, float)):
        raise ValueError("model run manifest threshold must be numeric")
    threshold = _canonical_threshold(float(manifest_threshold))
    sources = {
        ImageEncoderName.MERLIN.value: _load_embedding_run(
            Path(merlin_run), ImageEncoderName.MERLIN.value
        ),
        ImageEncoderName.SPECTRE.value: _load_embedding_run(
            Path(spectre_run), ImageEncoderName.SPECTRE.value
        ),
    }
    loaded_models = {
        encoder: _load_training_model(model_run, manifest, encoder, threshold)
        for encoder in (ImageEncoderName.MERLIN.value, ImageEncoderName.SPECTRE.value)
    }
    result_rows: list[dict[str, Any]] = []
    for encoder in (ImageEncoderName.MERLIN.value, ImageEncoderName.SPECTRE.value):
        source = sources[encoder]
        model = loaded_models[encoder][0]
        logits, probabilities, predictions = _infer(model, source.embeddings)
        result_rows.extend(
            _rows(
                encoder,
                source.patient_ids.tolist(),
                None,
                logits,
                probabilities,
                predictions,
            )
        )
    destination = _destination(Path(output_root), run_dir)
    write_summary(result_rows, destination / "classification_results.csv", PREDICTION_COLUMNS)
    write_json(
        {
            "pipeline_version": __version__,
            "stage": "recurrence_classification_prediction",
            "status": "complete",
            "model_run": str(model_run.resolve()),
            "model_run_manifest_sha256": sha256_file(manifest_path),
            "threshold": float(threshold),
            "sources": {
                encoder: {
                    "run": str(source.run.resolve()),
                    "image_embeddings_sha256": source.embedding_sha256,
                    "run_manifest_sha256": source.manifest_sha256,
                    "patient_count": int(source.patient_ids.size),
                }
                for encoder, source in sources.items()
            },
            "models": {
                encoder: {
                    "artifact": f"models/{encoder}.npz",
                    "sha256": digest,
                }
                for encoder, (_, digest) in loaded_models.items()
            },
            "artifacts": {"classification_results": "classification_results.csv"},
            "provisional_research_output": True,
        },
        destination / "run_manifest.json",
    )
    return destination
