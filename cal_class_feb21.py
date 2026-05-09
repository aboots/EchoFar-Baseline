from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class ConceptMetrics:
    concept_name: str
    n_samples: int
    n_ground_truth_non_null: int
    n_prediction_non_null: int
    n_correct: int
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    auroc_macro_ovr: float
    n_classes_ground_truth: int
    classes_ground_truth: Tuple[str, ...]


@dataclass(frozen=True)
class OverallMetrics:
    n_samples_total: int
    n_correct_total: int
    accuracy_micro: float
    balanced_accuracy_flat: float
    f1_macro_flat: float
    auroc_macro_ovr_flat: float
    balanced_accuracy_concept_weighted: float
    f1_macro_concept_weighted: float
    auroc_macro_ovr_concept_weighted: float


LABEL_TOKEN_CANONICALIZATION: Dict[str, str] = {
    "mildly": "mild",
    "moderately": "moderate",
    "severely": "severe",
    "severly": "severe",
    "serve": "severe",
    "servely": "severe",
    "trace": "trivial",
    "trivially": "trivial",
}

SEVERITY_TOKENS = {"trivial", "mild", "moderate", "severe"}
EXCLUDED_CONCEPT_COLUMNS: set[str] = {
    "PulmonaryValveArtery_pulmonary_artery_systolic_pressure",
    "Aorta_sinuses_of_valsalva"
}

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI args.

    Returns:
        argparse.ArgumentParser: Parser.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate concept-level classification CSVs (ground-truth vs predicted)."
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        default="/home/anne/report_gen/ground_truth_data.csv",
        help="Path to ground-truth CSV.",
    )
    parser.add_argument(
        "--predicted-csv",
        type=str,
        default="/home/anne/report_gen/findings_result_csv_feb20_echoprime.csv",
        help="Path to predicted CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/anne/report_gen/classification_eval_final_feb20",
        help="Directory to write evaluation outputs.",
    )
    parser.add_argument(
        "--missing-label",
        type=str,
        default="null",
        help="Canonical label used for missing values (e.g., 'null').",
    )
    parser.add_argument(
        "--export-aligned-csv",
        action="store_true",
        help="If set, export an aligned wide CSV with normalized GT/pred columns.",
    )
    return parser


def read_csv_as_strings(csv_path: Path) -> pd.DataFrame:
    """Read a CSV with all columns as strings.

    Args:
        csv_path (Path): CSV path.

    Returns:
        pd.DataFrame: Dataframe.
    """
    dataframe = pd.read_csv(csv_path, dtype="string", keep_default_na=True)
    return dataframe


def detect_id_column(dataframe: pd.DataFrame) -> str:
    """Detect the study ID column name.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        str: Column name ("exam_id" or "echo_id").

    Raises:
        ValueError: If neither ID column exists.
    """
    if "exam_id" in dataframe.columns:
        return "exam_id"
    if "echo_id" in dataframe.columns:
        return "echo_id"
    raise ValueError("Expected an 'exam_id' or 'echo_id' column in the CSV.")


def normalize_key_column(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    """Normalize a key column to stable strings for joining.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        column_name (str): Column name.

    Returns:
        pd.Series: Normalized series.
    """
    series = dataframe[column_name] if column_name in dataframe.columns else pd.Series([])
    series = series.astype("string").fillna("")
    series = series.str.strip()
    return series


def normalize_label(raw_value: Any, missing_label: str) -> str:
    """Normalize a label value to a canonical string.

    - Case-insensitive (e.g. "Normal" -> "normal")
    - Unifies separators (spaces, hyphens, slashes -> underscores)
    - Canonicalizes common severity tokens (e.g. "moderately" -> "moderate")
    - Treats empty/NaN/"null"/"nan"/"none"/"na"/"n/a" as missing_label

    Args:
        raw_value (Any): Raw cell value.
        missing_label (str): Canonical missing label.

    Returns:
        str: Normalized label string.
    """
    if raw_value is None:
        return missing_label

    if isinstance(raw_value, float) and math.isnan(raw_value):
        return missing_label

    text = str(raw_value).strip()
    if not text:
        return missing_label

    lowered = text.lower().strip()
    if lowered in {"null", "nan", "none", "na", "n/a"}:
        return missing_label

    lowered = lowered.replace("\u2013", "-").replace("\u2014", "-")
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = lowered.replace("/", " ").replace("-", " ")
    lowered = lowered.replace(" ", "_")
    lowered = re.sub(r"[^a-z0-9_]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")

    tokens = [token for token in lowered.split("_") if token]
    canonical_tokens = [LABEL_TOKEN_CANONICALIZATION.get(token, token) for token in tokens]
    normalized = "_".join(canonical_tokens).strip("_")

    if not normalized or normalized in {"null", "nan", "none", "na", "n/a"}:
        return missing_label

    return normalized


def split_severity_and_base(label: str) -> Tuple[Optional[str], str]:
    """Split a normalized label into (severity_token, base_without_severity).

    Examples:
        "moderate" -> ("moderate", "")
        "moderate_depressed" -> ("moderate", "depressed")
        "not_well_visualized" -> (None, "not_well_visualized")

    Args:
        label (str): Normalized label.

    Returns:
        Tuple[Optional[str], str]: Severity token (or None) and remaining base.
    """
    tokens = [token for token in label.split("_") if token]
    severity = next((token for token in tokens if token in SEVERITY_TOKENS), None)
    base_tokens = [token for token in tokens if token not in SEVERITY_TOKENS]
    base = "_".join(base_tokens)
    return severity, base


def apply_moderate_vs_severe_leniency(
    ground_truth_label: str,
    predicted_label: str,
) -> str:
    """Apply the requested leniency:
    If predicted is severe (including serve/servely normalized) and GT is moderate,
    treat it as correct (only when the non-severity base matches).

    Args:
        ground_truth_label (str): Normalized GT label.
        predicted_label (str): Normalized predicted label.

    Returns:
        str: Possibly-adjusted predicted label.
    """
    if ground_truth_label == predicted_label:
        return predicted_label

    gt_severity, gt_base = split_severity_and_base(ground_truth_label)
    pred_severity, pred_base = split_severity_and_base(predicted_label)

    if gt_severity == "moderate" and pred_severity == "severe" and gt_base == pred_base:
        return ground_truth_label

    return predicted_label


def build_concept_columns(
    ground_truth_df: pd.DataFrame,
    id_column: str,
) -> List[str]:
    """Extract concept columns from the ground-truth dataframe.

    Args:
        ground_truth_df (pd.DataFrame): Ground-truth dataframe.
        id_column (str): Study ID column name.

    Returns:
        List[str]: Concept column names (excluding metadata and excluded concepts).
    """
    metadata_columns = {id_column, "echo_id", "exam_id", "stream_id"}
    excluded_columns = metadata_columns.union(EXCLUDED_CONCEPT_COLUMNS)
    concept_columns = [col for col in ground_truth_df.columns if col not in excluded_columns]
    return concept_columns


def safe_balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute balanced accuracy safely.

    Args:
        y_true (Sequence[str]): True labels.
        y_pred (Sequence[str]): Predicted labels.

    Returns:
        float: Balanced accuracy, or NaN if not computable.
    """
    if not y_true:
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(y_true, y_pred))


def safe_f1_macro(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute macro F1 safely.

    Args:
        y_true (Sequence[str]): True labels.
        y_pred (Sequence[str]): Predicted labels.

    Returns:
        float: Macro F1, or NaN if not computable.
    """
    if not y_true:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def safe_auroc_macro_ovr(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute AUROC (macro, one-vs-rest) using hard predictions as scores.

    For binary:
        Uses score = 1 if predicted == positive_class else 0
    For multiclass:
        Uses one-hot hard predictions as class scores.

    Args:
        y_true (Sequence[str]): True labels.
        y_pred (Sequence[str]): Predicted labels.

    Returns:
        float: AUROC, or NaN if undefined (e.g. only one class in y_true).
    """
    if not y_true:
        return float("nan")

    unique_true = sorted(set(y_true))
    if len(unique_true) < 2:
        return float("nan")

    try:
        if len(unique_true) == 2:
            positive_class = unique_true[1]
            y_true_binary = np.array([1 if label == positive_class else 0 for label in y_true])
            y_score_binary = np.array([1 if label == positive_class else 0 for label in y_pred])
            return float(roc_auc_score(y_true_binary, y_score_binary))

        classes = unique_true
        class_to_index = {label: idx for idx, label in enumerate(classes)}
        y_score = np.zeros((len(y_pred), len(classes)), dtype=float)
        for row_index, pred_label in enumerate(y_pred):
            column_index = class_to_index.get(pred_label)
            if column_index is not None:
                y_score[row_index, column_index] = 1.0

        return float(
            roc_auc_score(
                np.array(y_true),
                y_score,
                average="macro",
                multi_class="ovr",
                labels=np.array(classes),
            )
        )
    except ValueError:
        return float("nan")


def compute_concept_metrics(
    concept_name: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    missing_label: str,
) -> ConceptMetrics:
    """Compute metrics for a single concept column.

    Any row where either GT or prediction equals `missing_label` is ignored
    (not counted as correct or wrong, and not included in metric denominators).

    Args:
        concept_name (str): Concept column name.
        y_true (Sequence[str]): Normalized ground-truth labels.
        y_pred (Sequence[str]): Normalized predicted labels (after leniency).
        missing_label (str): Canonical missing label.

    Returns:
        ConceptMetrics: Metrics.
    """
    y_true_all = list(y_true)
    y_pred_all = list(y_pred)

    n_ground_truth_non_null = int(sum(1 for gt in y_true_all if gt != missing_label))
    n_prediction_non_null = int(sum(1 for pr in y_pred_all if pr != missing_label))

    valid_pairs = [
        (gt, pr)
        for gt, pr in zip(y_true_all, y_pred_all)
        if gt != missing_label and pr != missing_label
    ]

    y_true_valid = [gt for gt, _ in valid_pairs]
    y_pred_valid = [pr for _, pr in valid_pairs]

    n_samples = len(valid_pairs)
    n_correct = int(sum(1 for gt, pr in valid_pairs if gt == pr))

    accuracy = float(n_correct / n_samples) if n_samples else float("nan")
    balanced_accuracy = safe_balanced_accuracy(y_true_valid, y_pred_valid)
    f1_macro = safe_f1_macro(y_true_valid, y_pred_valid)
    auroc_macro_ovr = safe_auroc_macro_ovr(y_true_valid, y_pred_valid)

    classes_ground_truth = tuple(sorted(set(gt for gt in y_true_all if gt != missing_label)))

    return ConceptMetrics(
        concept_name=concept_name,
        n_samples=n_samples,
        n_ground_truth_non_null=n_ground_truth_non_null,
        n_prediction_non_null=n_prediction_non_null,
        n_correct=n_correct,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        f1_macro=f1_macro,
        auroc_macro_ovr=auroc_macro_ovr,
        n_classes_ground_truth=len(classes_ground_truth),
        classes_ground_truth=classes_ground_truth,
    )

def weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    """Compute a weighted mean, skipping NaNs.

    Args:
        values (Sequence[float]): Values.
        weights (Sequence[int]): Weights.

    Returns:
        float: Weighted mean, or NaN if all values are NaN / total weight is 0.
    """
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights)
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if not pairs:
        return float("nan")

    numerator = float(sum(value * weight for value, weight in pairs))
    denominator = float(sum(weight for _, weight in pairs))
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def evaluate(
    ground_truth_csv: Path,
    predicted_csv: Path,
    output_dir: Path,
    missing_label: str,
    export_aligned_csv: bool,
) -> Tuple[OverallMetrics, List[ConceptMetrics]]:
    """Run evaluation.

    Args:
        ground_truth_csv (Path): Path to GT CSV.
        predicted_csv (Path): Path to predicted CSV.
        output_dir (Path): Output directory.
        missing_label (str): Canonical missing label.
        export_aligned_csv (bool): Whether to export aligned normalized CSV.

    Returns:
        Tuple[OverallMetrics, List[ConceptMetrics]]: Overall + per-concept metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_df = read_csv_as_strings(ground_truth_csv)
    predicted_df = read_csv_as_strings(predicted_csv)

    ground_truth_id_column = detect_id_column(ground_truth_df)
    predicted_id_column = detect_id_column(predicted_df)

    if predicted_id_column != ground_truth_id_column:
        predicted_df = predicted_df.rename(columns={predicted_id_column: ground_truth_id_column})

    id_column = ground_truth_id_column

    ground_truth_df[id_column] = normalize_key_column(ground_truth_df, id_column)
    predicted_df[id_column] = normalize_key_column(predicted_df, id_column)

    join_columns = [id_column]
    should_use_stream_id = (
        "stream_id" in ground_truth_df.columns and "stream_id" in predicted_df.columns
    )
    if should_use_stream_id:
        duplicated_by_id = bool(ground_truth_df.duplicated([id_column]).any())
        if duplicated_by_id:
            ground_truth_df["stream_id"] = normalize_key_column(ground_truth_df, "stream_id")
            predicted_df["stream_id"] = normalize_key_column(predicted_df, "stream_id")
            join_columns = [id_column, "stream_id"]

    ground_truth_df = ground_truth_df.drop_duplicates(subset=join_columns, keep="first")
    predicted_df = predicted_df.drop_duplicates(subset=join_columns, keep="first")

    concept_columns = build_concept_columns(ground_truth_df, id_column=id_column)

    ground_truth_indexed = ground_truth_df.set_index(join_columns, drop=False)
    predicted_indexed = predicted_df.set_index(join_columns, drop=False)
    predicted_aligned = predicted_indexed.reindex(ground_truth_indexed.index)

    ground_truth_concepts = ground_truth_indexed.reindex(columns=concept_columns)
    predicted_concepts = predicted_aligned.reindex(columns=concept_columns)

    per_concept_metrics: List[ConceptMetrics] = []
    y_true_flat: List[str] = []
    y_pred_flat: List[str] = []

    aligned_export_rows: Optional[pd.DataFrame] = None
    if export_aligned_csv:
        aligned_export_rows = ground_truth_indexed[join_columns].copy()

    for concept_name in concept_columns:
        gt_series = ground_truth_concepts[concept_name].map(
            lambda value: normalize_label(value, missing_label=missing_label)
        )
        pred_series = predicted_concepts[concept_name].map(
            lambda value: normalize_label(value, missing_label=missing_label)
        )

        y_true = gt_series.to_list()
        y_pred_raw = pred_series.to_list()
        y_pred = [
            apply_moderate_vs_severe_leniency(gt_label, pred_label)
            for gt_label, pred_label in zip(y_true, y_pred_raw)
        ]

        per_concept_metrics.append(
            compute_concept_metrics(
                concept_name=concept_name,
                y_true=y_true,
                y_pred=y_pred,
                missing_label=missing_label,
            )
        )

        y_true_flat.extend(y_true)
        y_pred_flat.extend(y_pred)

        if aligned_export_rows is not None:
            aligned_export_rows[f"{concept_name}__gt"] = gt_series
            aligned_export_rows[f"{concept_name}__pred"] = pd.Series(y_pred, index=gt_series.index)

    overall_flat_metrics = compute_concept_metrics(
        concept_name="__overall_flat__",
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        missing_label=missing_label,
    )

    concept_weights = [metrics.n_samples for metrics in per_concept_metrics]

    overall_metrics = OverallMetrics(
        n_samples_total=int(sum(metrics.n_samples for metrics in per_concept_metrics)),
        n_correct_total=int(sum(metrics.n_correct for metrics in per_concept_metrics)),
        accuracy_micro=float(
            (sum(metrics.n_correct for metrics in per_concept_metrics) / sum(concept_weights))
            if sum(concept_weights) > 0
            else float("nan")
        ),
        balanced_accuracy_flat=float(overall_flat_metrics.balanced_accuracy),
        f1_macro_flat=float(overall_flat_metrics.f1_macro),
        auroc_macro_ovr_flat=float(overall_flat_metrics.auroc_macro_ovr),
        balanced_accuracy_concept_weighted=weighted_mean(
            [metrics.balanced_accuracy for metrics in per_concept_metrics],
            concept_weights,
        ),
        f1_macro_concept_weighted=weighted_mean(
            [metrics.f1_macro for metrics in per_concept_metrics],
            concept_weights,
        ),
        auroc_macro_ovr_concept_weighted=weighted_mean(
            [metrics.auroc_macro_ovr for metrics in per_concept_metrics],
            concept_weights,
        ),
    )

    per_concept_df = pd.DataFrame([asdict(metrics) for metrics in per_concept_metrics])
    per_concept_df["classes_ground_truth"] = per_concept_df["classes_ground_truth"].map(
        lambda classes: "|".join(classes) if isinstance(classes, (list, tuple)) else str(classes)
    )

    per_concept_csv_path = output_dir / "per_concept_metrics.csv"
    per_concept_df.to_csv(per_concept_csv_path, index=False)

    overall_json_path = output_dir / "overall_metrics.json"
    overall_json_path.write_text(
        json.dumps(asdict(overall_metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if aligned_export_rows is not None:
        aligned_csv_path = output_dir / "aligned_normalized_gt_pred.csv"
        aligned_export_rows.to_csv(aligned_csv_path, index=False)

    return overall_metrics, per_concept_metrics


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    overall_metrics, _ = evaluate(
        ground_truth_csv=Path(args.ground_truth_csv),
        predicted_csv=Path(args.predicted_csv),
        output_dir=Path(args.output_dir),
        missing_label=str(args.missing_label).strip().lower() or "null",
        export_aligned_csv=bool(args.export_aligned_csv),
    )

    print(json.dumps(asdict(overall_metrics), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
