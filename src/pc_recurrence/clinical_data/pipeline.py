from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pc_recurrence import __version__
from pc_recurrence.io import (
    create_run_directory,
    sha256_file,
    write_json,
    write_summary,
)

from .constants import DEFAULT_OUTPUT_ROOT
from .workbook import (
    ClinicalWorkbookRow,
    load_clinical_workbook,
    select_clinical_workbook_rows,
)

ProgressReporter = Callable[[str], None]

MISSING_CATEGORY = "__missing__"
UNKNOWN_CATEGORY = "__unknown__"
SCALE_FLOOR = 1e-6
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 0


@dataclass(frozen=True)
class NumericField:
    attribute: str
    source_header: str
    output_name: str


@dataclass(frozen=True)
class CategoricalField:
    attribute: str
    source_header: str
    output_name: str
    multi_value: bool = False


@dataclass(frozen=True)
class ParsedNumber:
    value: float | None
    below_detection_limit: bool = False


NUMERIC_FIELDS = (
    NumericField("age", "yaş", "age"),
    NumericField("ca_19_9", "CA 19-9", "ca_19_9"),
    NumericField("total_bilirubin", "total bilirubin", "total_bilirubin"),
    NumericField("direct_bilirubin", "direkt bilirubin", "direct_bilirubin"),
    NumericField("serum_albumin", "serum albumin", "serum_albumin"),
    NumericField("absolute_lymphocyte", "mutlak lenfosit", "absolute_lymphocyte"),
    NumericField("crp", "CRP", "crp"),
    NumericField("cally", "CALLY", "cally"),
    NumericField("pni", "PNI", "pni"),
)

CATEGORICAL_FIELDS = (
    CategoricalField("surgery", "ameliyat şekli", "surgery"),
    CategoricalField("pathology", "patoloji", "pathology"),
    CategoricalField("symptom", "semptom", "symptom", multi_value=True),
)

_BOUND_PATTERN = re.compile(r"^<\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))$")
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)$")
_SLUG_TRANSLATION = str.maketrans(
    {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"}
)


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(f"[pc-clinical-data] {message}")


def _parse_number(value: Any, *, field: NumericField, row: ClinicalWorkbookRow) -> ParsedNumber:
    if value is None:
        return ParsedNumber(None)
    if isinstance(value, bool):
        raise ValueError(
            f"Row {row.row_number} patient {row.patient_id!r} field "
            f"{field.source_header!r} contains a Boolean instead of a number"
        )
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"Row {row.row_number} patient {row.patient_id!r} field "
                f"{field.source_header!r} is not finite"
            )
        return ParsedNumber(number)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() == "yok":
            return ParsedNumber(None)
        bound_match = _BOUND_PATTERN.fullmatch(text)
        if bound_match:
            number = float(bound_match.group(1).replace(",", "."))
            if not math.isfinite(number):
                raise ValueError(
                    f"Row {row.row_number} patient {row.patient_id!r} field "
                    f"{field.source_header!r} has a non-finite detection bound"
                )
            return ParsedNumber(number, below_detection_limit=True)
        if _NUMBER_PATTERN.fullmatch(text):
            return ParsedNumber(float(text.replace(",", ".")))
    raise ValueError(
        f"Row {row.row_number} patient {row.patient_id!r} field "
        f"{field.source_header!r} has unsupported numeric value {value!r}"
    )


def _canonical_category(value: Any) -> str:
    if value is None:
        return MISSING_CATEGORY
    return " ".join(str(value).strip().casefold().split()) or MISSING_CATEGORY


def _category_values(row: ClinicalWorkbookRow, field: CategoricalField) -> set[str]:
    value = getattr(row, field.attribute)
    if not field.multi_value:
        return {_canonical_category(value)}
    if value is None:
        return {MISSING_CATEGORY}
    values = {
        _canonical_category(part)
        for part in str(value).split(",")
        if str(part).strip()
    }
    return values or {MISSING_CATEGORY}


def _slug(value: str) -> str:
    if value == MISSING_CATEGORY:
        return "missing"
    if value == UNKNOWN_CATEGORY:
        return "unknown"
    transliterated = value.translate(_SLUG_TRANSLATION)
    ascii_text = unicodedata.normalize("NFKD", transliterated).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Category {value!r} cannot be represented as a feature name")
    return slug


def _category_columns(field: CategoricalField, vocabulary: list[str]) -> dict[str, str]:
    categories = [*vocabulary, MISSING_CATEGORY, UNKNOWN_CATEGORY]
    mapping = {category: f"{field.output_name}__{_slug(category)}" for category in categories}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            f"Categories for {field.source_header!r} collide after feature-name normalization"
        )
    return mapping


def _parse_target(value: Any, *, patient_id: str) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Patient {patient_id!r} has a blank recurrence label")
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        if value in (0, 1):
            return int(value)
        raise ValueError(f"Patient {patient_id!r} has invalid binary recurrence label {value!r}")
    if isinstance(value, str) and value.strip().casefold() in {"yok", "0"}:
        return 0
    if isinstance(value, str) and value.strip() == "1":
        return 1
    return 1


def _split_rows(
    rows: list[ClinicalWorkbookRow],
    targets: dict[str, int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ClinicalWorkbookRow], list[ClinicalWorkbookRow]]:
    """Create a deterministic patient-level split, stratifying when both classes permit it."""
    if len(rows) < 2:
        raise ValueError("at least two patients are required to create train and validation tables")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and strictly between 0 and 1")
    if seed < 0:
        raise ValueError("split seed must be non-negative")

    validation_count = min(len(rows) - 1, max(1, math.ceil(validation_fraction * len(rows))))
    indices = np.arange(len(rows), dtype=np.int64)
    positive = np.asarray(
        [index for index, row in enumerate(rows) if targets[row.patient_id] == 1],
        dtype=np.int64,
    )
    negative = np.asarray(
        [index for index, row in enumerate(rows) if targets[row.patient_id] == 0],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    if (
        validation_count >= 2
        and positive.size >= 2
        and negative.size >= 2
    ):
        validation_positive_count = int(round(validation_count * positive.size / len(rows)))
        validation_positive_count = max(1, min(positive.size - 1, validation_positive_count))
        validation_negative_count = validation_count - validation_positive_count
        if validation_negative_count < 1:
            validation_positive_count -= 1
            validation_negative_count += 1
        if validation_negative_count >= negative.size:
            validation_positive_count += validation_negative_count - (negative.size - 1)
            validation_negative_count = negative.size - 1
        chosen = np.concatenate(
            [
                rng.choice(positive, size=validation_positive_count, replace=False),
                rng.choice(negative, size=validation_negative_count, replace=False),
            ]
        )
    else:
        chosen = rng.choice(indices, size=validation_count, replace=False)
    validation_indices = set(int(index) for index in chosen.tolist())
    train_rows = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, validation_rows


def _destination(output_root: Path, run_dir: Path | None) -> Path:
    if run_dir is None:
        return create_run_directory(output_root)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def _raw_columns() -> list[str]:
    return [
        "patient_id",
        "target_recurrence",
        "ct_report_text",
        "ct_report_missing",
        *(field.output_name for field in NUMERIC_FIELDS),
        *(field.output_name for field in CATEGORICAL_FIELDS),
        "ca_19_9_below_detection_limit",
    ]


def _raw_record(
    row: ClinicalWorkbookRow,
    targets: dict[str, int],
    parsed_numbers: dict[str, dict[str, ParsedNumber]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "patient_id": row.patient_id,
        "target_recurrence": targets[row.patient_id],
        "ct_report_text": "" if row.ct_report is None else str(row.ct_report).strip(),
        "ct_report_missing": int(row.ct_report is None),
    }
    for field in NUMERIC_FIELDS:
        value = getattr(row, field.attribute)
        record[field.output_name] = value
    for field in CATEGORICAL_FIELDS:
        value = getattr(row, field.attribute)
        record[field.output_name] = "" if value is None else str(value).strip()
    record["ca_19_9_below_detection_limit"] = int(
        parsed_numbers["ca_19_9"][row.patient_id].below_detection_limit
    )
    return record


def preprocess_clinical_features(
    workbook_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    run_dir: Path | None = None,
    patients: set[str] | None = None,
    fit_patients: set[str] | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    progress: ProgressReporter | None = None,
) -> Path:
    """Create auditable train/validation clinical tables and preserve CT report text."""
    workbook_path = Path(workbook_path)
    _report(progress, f"Loading clinical variables from {workbook_path}.")
    source_rows = load_clinical_workbook(workbook_path)
    rows = select_clinical_workbook_rows(source_rows, patients)
    if not rows:
        raise ValueError("the selected clinical cohort is empty")
    targets = {
        row.patient_id: _parse_target(row.recurrence_raw, patient_id=row.patient_id)
        for row in rows
    }
    parsed_numbers: dict[str, dict[str, ParsedNumber]] = {
        field.attribute: {
            row.patient_id: _parse_number(getattr(row, field.attribute), field=field, row=row)
            for row in rows
        }
        for field in NUMERIC_FIELDS
    }
    train_rows, validation_rows = _split_rows(rows, targets, validation_fraction, split_seed)
    fit_rows = (
        train_rows
        if fit_patients is None
        else select_clinical_workbook_rows(train_rows, fit_patients)
    )
    if not fit_rows:
        raise ValueError("the preprocessing fit cohort is empty")
    _report(
        progress,
        f"Split {len(rows)} patient(s) into {len(train_rows)} train and "
        f"{len(validation_rows)} validation patient(s).",
    )
    _report(
        progress,
        f"Fitting preprocessing on {len(fit_rows)} training patient(s) and transforming "
        "both tables.",
    )

    numeric_parameters: dict[str, dict[str, Any]] = {}
    for field in NUMERIC_FIELDS:
        fit_values = [
            parsed_numbers[field.attribute][row.patient_id].value
            for row in fit_rows
            if parsed_numbers[field.attribute][row.patient_id].value is not None
        ]
        if not fit_values:
            raise ValueError(
                f"Numeric field {field.source_header!r} has no observed values in the fit cohort"
            )
        mean = float(np.mean(np.asarray(fit_values, dtype=np.float64)))
        raw_scale = float(np.std(np.asarray(fit_values, dtype=np.float64)))
        scale = 1.0 if raw_scale < SCALE_FLOOR else raw_scale
        numeric_parameters[field.attribute] = {
            "source_header": field.source_header,
            "normalized_column": f"{field.output_name}_zscore",
            "missing_indicator_column": f"{field.output_name}_missing",
            "mean_imputation_value": mean,
            "mean": mean,
            "population_standard_deviation": raw_scale,
            "scale": scale,
            "scale_floor": SCALE_FLOOR,
            "observed_fit_count": len(fit_values),
            "missing_fit_count": len(fit_rows) - len(fit_values),
        }

    categorical_parameters: dict[str, dict[str, Any]] = {}
    categorical_columns: dict[str, dict[str, str]] = {}
    for field in CATEGORICAL_FIELDS:
        vocabulary = sorted(
            {
                category
                for row in fit_rows
                for category in _category_values(row, field)
                if category not in {MISSING_CATEGORY, UNKNOWN_CATEGORY}
            }
        )
        columns = _category_columns(field, vocabulary)
        categorical_columns[field.attribute] = columns
        categorical_parameters[field.attribute] = {
            "source_header": field.source_header,
            "multi_value": field.multi_value,
            "separator": "," if field.multi_value else None,
            "vocabulary": vocabulary,
            "columns": columns,
        }

    numeric_feature_columns = [
        f"{field.output_name}_zscore" for field in NUMERIC_FIELDS
    ] + [f"{field.output_name}_missing" for field in NUMERIC_FIELDS]
    numeric_feature_columns.append("ca_19_9_below_detection_limit")
    category_feature_columns = [
        column
        for field in CATEGORICAL_FIELDS
        for column in categorical_columns[field.attribute].values()
    ]
    feature_columns = [*numeric_feature_columns, *category_feature_columns]
    feature_columns_with_metadata = [
        "patient_id",
        "target_recurrence",
        "ct_report_text",
        "ct_report_missing",
        *feature_columns,
    ]

    output_rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        record: dict[str, Any] = {
            "patient_id": row.patient_id,
            "target_recurrence": targets[row.patient_id],
            "ct_report_text": "" if row.ct_report is None else str(row.ct_report).strip(),
            "ct_report_missing": int(row.ct_report is None),
        }
        for field in NUMERIC_FIELDS:
            parsed = parsed_numbers[field.attribute][row.patient_id]
            parameters = numeric_parameters[field.attribute]
            value = (
                parameters["mean_imputation_value"]
                if parsed.value is None
                else parsed.value
            )
            record[f"{field.output_name}_zscore"] = (
                value - parameters["mean"]
            ) / parameters["scale"]
            record[f"{field.output_name}_missing"] = int(parsed.value is None)
        record["ca_19_9_below_detection_limit"] = int(
            parsed_numbers["ca_19_9"][row.patient_id].below_detection_limit
        )
        for field in CATEGORICAL_FIELDS:
            source_values = _category_values(row, field)
            mapping = categorical_columns[field.attribute]
            encoded_values = {
                category if category in mapping else UNKNOWN_CATEGORY
                for category in source_values
            }
            for category, column in mapping.items():
                record[column] = int(category in encoded_values)
        output_rows_by_id[row.patient_id] = record

    destination = _destination(Path(output_root), run_dir)
    raw_columns = _raw_columns()
    raw_rows_by_id = {
        row.patient_id: _raw_record(row, targets, parsed_numbers) for row in rows
    }
    train_ids = [row.patient_id for row in train_rows]
    validation_ids = [row.patient_id for row in validation_rows]
    train_id_set = set(train_ids)
    raw_audit_columns = ["patient_id", "split", *raw_columns[1:]]
    raw_audit_path = write_summary(
        [
            {
                **raw_rows_by_id[patient_id],
                "split": "train" if patient_id in train_id_set else "validation",
            }
            for patient_id in [*train_ids, *validation_ids]
        ],
        destination / "clinical_raw.csv",
        tuple(raw_audit_columns),
    )
    train_feature_path = write_summary(
        [output_rows_by_id[patient_id] for patient_id in train_ids],
        destination / "clinical_features_train.csv",
        tuple(feature_columns_with_metadata),
    )
    validation_feature_path = write_summary(
        [output_rows_by_id[patient_id] for patient_id in validation_ids],
        destination / "clinical_features_validation.csv",
        tuple(feature_columns_with_metadata),
    )
    train_raw_path = write_summary(
        [raw_rows_by_id[patient_id] for patient_id in train_ids],
        destination / "clinical_raw_train.csv",
        tuple(raw_columns),
    )
    validation_raw_path = write_summary(
        [raw_rows_by_id[patient_id] for patient_id in validation_ids],
        destination / "clinical_raw_validation.csv",
        tuple(raw_columns),
    )
    split_assignments_path = write_summary(
        [
            {
                "patient_id": patient_id,
                "split": "train" if patient_id in set(train_ids) else "validation",
                "target_recurrence": targets[patient_id],
                "source_row_number": next(
                    row.row_number for row in rows if row.patient_id == patient_id
                ),
            }
            for patient_id in [*train_ids, *validation_ids]
        ],
        destination / "split_assignments.csv",
        ("patient_id", "split", "target_recurrence", "source_row_number"),
    )
    parameters_path = write_json(
        {
            "schema_version": 1,
            "fit_patient_ids": [row.patient_id for row in fit_rows],
            "fit_scope": "training_cohort" if fit_patients is None else "explicit_training_subset",
            "split": {
                "strategy": "deterministic patient-level random split",
                "validation_fraction": validation_fraction,
                "seed": split_seed,
                "train_patient_ids": train_ids,
                "validation_patient_ids": validation_ids,
            },
            "normalization": "z-score using fit-cohort population standard deviation",
            "numeric_missing_value_policy": "fit-cohort mean imputation plus missingness indicator",
            "numeric_fields": numeric_parameters,
            "ca_19_9_below_detection_policy": {
                "input_syntax": "<limit",
                "numeric_value": "recorded upper bound",
                "indicator_column": "ca_19_9_below_detection_limit",
            },
            "categorical_encoding": "one-hot",
            "categorical_fields": categorical_parameters,
            "feature_columns": feature_columns,
            "non_feature_columns": [
                "patient_id",
                "target_recurrence",
                "ct_report_text",
                "ct_report_missing",
            ],
            "text_embedding": {
                "status": "pending",
                "source_column": "ct_report_text",
                "model": None,
            },
        },
        destination / "preprocessing_parameters.json",
    )
    def _artifact(path: Path) -> dict[str, str]:
        return {"path": path.name, "sha256": sha256_file(path)}

    write_json(
        {
            "pipeline_version": __version__,
            "stage": "clinical_feature_preprocessing",
            "status": "complete",
            "workbook": str(workbook_path.resolve()),
            "workbook_sha256": sha256_file(workbook_path),
            "patient_count": len(rows),
            "fit_patient_count": len(fit_rows),
            "train_patient_count": len(train_rows),
            "validation_patient_count": len(validation_rows),
            "patient_ids": [row.patient_id for row in rows],
            "fit_patient_ids": [row.patient_id for row in fit_rows],
            "split": {
                "strategy": "deterministic patient-level random split",
                "validation_fraction": validation_fraction,
                "seed": split_seed,
                "train_patient_ids": train_ids,
                "validation_patient_ids": validation_ids,
                "train_class_counts": {
                    "positive": sum(targets[patient_id] for patient_id in train_ids),
                    "negative": sum(1 - targets[patient_id] for patient_id in train_ids),
                },
                "validation_class_counts": {
                    "positive": sum(targets[patient_id] for patient_id in validation_ids),
                    "negative": sum(1 - targets[patient_id] for patient_id in validation_ids),
                },
            },
            "feature_count": len(feature_columns),
            "target_column": "target_recurrence",
            "join_key": "patient_id",
            "ct_report_column": "ct_report_text",
            "excluded_source_columns": {
                "hasta no": "anonymized source/DICOM locator; identifier, not a feature",
                "Görüntü alanı": "CT slice-selection metadata, not a clinical feature",
                "nüks": "converted to target_recurrence; never included as an input feature",
            },
            "artifacts": {
                "clinical_features_train": _artifact(train_feature_path),
                "clinical_features_validation": _artifact(validation_feature_path),
                "clinical_raw": _artifact(raw_audit_path),
                "clinical_raw_train": _artifact(train_raw_path),
                "clinical_raw_validation": _artifact(validation_raw_path),
                "split_assignments": _artifact(split_assignments_path),
                "preprocessing_parameters": _artifact(parameters_path),
            },
            "provisional_research_output": True,
        },
        destination / "run_manifest.json",
    )
    _report(
        progress,
        f"Finished: {len(train_rows)} train + {len(validation_rows)} validation rows and "
        f"{len(feature_columns)} clinical features. "
        f"Artifacts: {destination}",
    )
    return destination
