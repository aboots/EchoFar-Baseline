from __future__ import annotations

import json
import re

import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import argparse
import csv
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from peft import PeftModel
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .attention_patch import *
from .common_doppler import *
from .data import *
from .logging_utils import *
from .metrics import *
from .model_doppler import *
from .torch_lm_utils import *
from .train_utils import *


DEFAULT_CONCEPT_CSV_PATH = Path("/home/anne/report_gen/findings_all_feb10_all_finalversion_balanced_not_visua.csv")

_ALLOWED_SUFFIXES = (".npy", ".npz", ".pt", ".pth")
_STRIPPED_KEYS = ("video_features", "video_mask", "patch_features", "embedding", "embeddings")


def _extract_exam_id(item: Mapping[str, Any]) -> str:
    for key in ("exam_id", "echo_id", "study_id", "id"):
        value = item.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError("Missing exam identifier (exam_id/echo_id/study_id/id).")


def _load_embedding_tensor(path: Union[str, Path]) -> torch.Tensor:
    resolved_path = Path(path)
    suffix = resolved_path.suffix.lower()

    if suffix == ".npy":
        array = np.load(str(resolved_path), mmap_mode="r")
        return torch.from_numpy(array)

    if suffix == ".npz":
        archive = np.load(str(resolved_path), mmap_mode="r")
        for key in ("video_features", "features"):
            if key in archive:
                return torch.from_numpy(archive[key])
        keys = list(archive.keys())
        if not keys:
            raise ValueError(f"Empty npz archive: {resolved_path}")
        return torch.from_numpy(archive[keys[0]])

    if suffix in {".pt", ".pth"}:
        payload = torch.load(str(resolved_path), map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, Mapping):
            for key in ("video_features", "features", "embedding", "embeddings"):
                tensor_value = payload.get(key, None)
                if isinstance(tensor_value, torch.Tensor):
                    return tensor_value
        raise ValueError(f"Unsupported torch payload in: {resolved_path}")

    raise ValueError(f"Unsupported embedding suffix: {resolved_path}")


def _resolve_embedding_paths(
    item: Mapping[str, Any],
    embedding_dir: Path,
    max_videos_per_study: int,
) -> Tuple[Path, ...]:
    list_keys = (
        "video_feature_paths",
        "video_features_paths",
        "patch_features_paths",
        "embedding_paths",
    )
    for key in list_keys:
        value = item.get(key, None)
        if isinstance(value, (list, tuple)):
            paths = [Path(str(v)) for v in value if v is not None and str(v).strip()]
            if paths:
                return tuple(paths[: max(1, int(max_videos_per_study))])

    single_keys = (
        "video_feature_path",
        "video_features_path",
        "patch_features_path",
        "embedding_path",
    )
    for key in single_keys:
        value = item.get(key, None)
        if value is not None and str(value).strip():
            return (Path(str(value)),)

    exam_id = _extract_exam_id(item)
    candidates: List[Path] = []
    exam_dir = embedding_dir / exam_id

    if exam_dir.is_dir():
        for suffix in _ALLOWED_SUFFIXES:
            candidates.extend(sorted(exam_dir.glob(f"*{suffix}")))
    else:
        for suffix in _ALLOWED_SUFFIXES:
            candidates.extend(sorted(embedding_dir.glob(f"{exam_id}*{suffix}")))

    if not candidates:
        raise FileNotFoundError(f"No embeddings for exam_id={exam_id} under {embedding_dir}")

    return tuple(candidates[: max(1, int(max_videos_per_study))])


class LazyEchoPrimeReportDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: Sequence[Mapping[str, Any]],
        embedding_dir: Union[str, Path],
        max_videos_per_study: int,
    ) -> None:
        self.embedding_dir = Path(embedding_dir)
        self.max_videos_per_study = int(max_videos_per_study)

        self.items: List[Dict[str, Any]] = []
        self.embedding_paths_by_index: List[Tuple[Path, ...]] = []

        for raw_item in items:
            if isinstance(raw_item, dict):
                item = raw_item
                for key in _STRIPPED_KEYS:
                    item.pop(key, None)
            else:
                for key in _STRIPPED_KEYS:
                    if hasattr(raw_item, key):
                        try:
                            setattr(raw_item, key, None)
                        except Exception:
                            pass
                item = dict(raw_item)

            embedding_paths = _resolve_embedding_paths(
                item=item,
                embedding_dir=self.embedding_dir,
                max_videos_per_study=self.max_videos_per_study,
            )

            self.items.append(item)
            self.embedding_paths_by_index.append(embedding_paths)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.items[int(index)])
        embedding_paths = self.embedding_paths_by_index[int(index)]

        tensors = [_load_embedding_tensor(path) for path in embedding_paths]
        video_features = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

        if video_features.ndim != 2:
            raise ValueError(f"video_features must be [tokens, dim], got {tuple(video_features.shape)}")

        item["video_features"] = video_features
        item["video_mask"] = torch.ones((int(video_features.shape[0]),), dtype=torch.bool)
        return item
        
CONCEPT_SPECS: Dict[str, Sequence[str]] = {
    "AorticValve_regurgitation": ["normal", "mild", "moderate"],
    "AorticValve_stenosis": ["normal", "mild", "moderate"],
    "AorticValve_valve_thickening": ["normal", "mild", "moderate"],
    "LeftAtrium_chamber_size": ["normal", "mild_dilated", "moderate_dilated"],
    "LeftVentricle_chamber_size": ["normal", "mild_dilated", "moderate_dilated"],
    "LeftVentricle_diastolic_function": [
        "normal",
        "mild_diastolic_dysfunction",
        "moderate_diastolic_dysfunction",
    ],
    "LeftVentricle_filling_pressure": ["normal", "mildly_elevated"],
    "LeftVentricle_systolic_function": [
        "normal",
        "mildly_depressed",
        "moderately_depressed",
    ],
    "LeftVentricle_wall_motion": ["normal", "abnormal"],
    "LeftVentricle_wall_thickness": [
        "normal",
        "abnormal",
        "concentric_remodeling",
    ],
    "MitralValve_leaflet_thickening": ["normal", "mild", "moderate"],
    "MitralValve_regurgitation": ["normal", "mild", "moderate"],
    "RightAtrium_chamber_size": ["normal", "mild_dilated", "moderate_dilated"],
    "RightVentricle_chamber_size": ["normal", "abnormal"],
    "RightVentricle_systolic_function": [
        "normal",
        "mildly_depressed",
        "moderately_depressed",
    ],
    "TricuspidValve_regurgitation": ["normal", "mild", "moderate"],
    "Aorta_ascending_aorta":["normal","dilated"],
    "Aorta_sinuses_of_valsalva":["normal","dilated"],
    "PericardiumOther_pericardial_effusion":["normal","abnormal"],
    "PulmonaryValveArtery_pulmonary_artery_systolic_pressure":["normal","elevated"],
    "PulmonaryValveArtery_regurgitation": ["normal", "mild", "moderate"],
    "Venous_inferior_vena_cava":["normal","dilated"],
}

CONCEPT_NAMES: Tuple[str, ...] = tuple(CONCEPT_SPECS.keys())
#batch_size_for_prompt = 128 ########TODO: CHANGE IT HERE 

def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _normalize_concept_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "none", "nan", ""}:
        return None

    normalized = re.sub(r"[\s\-]+", "_", lowered)
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    return normalized


def _concept_region(concept_name: str) -> str:
    return str(concept_name).split("_", 1)[0]


def _build_region_to_concept_indices(concept_names: Sequence[str]) -> Dict[str, List[int]]:
    region_to_indices: Dict[str, List[int]] = {}
    for i, name in enumerate(concept_names):
        region = _concept_region(name)
        region_to_indices.setdefault(region, []).append(int(i))
    return region_to_indices


REGION_TO_CONCEPT_INDICES = _build_region_to_concept_indices(CONCEPT_NAMES)


@dataclass(frozen=True)
class ConceptLabelTable:
    concept_names: Tuple[str, ...]
    label_to_index_by_concept: Dict[str, Dict[str, int]]
    exam_id_to_targets: Dict[str, Tuple[int, ...]]
    ignore_index: int

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        concept_specs: Mapping[str, Sequence[str]],
        ignore_index: int,
    ) -> "ConceptLabelTable":
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Concept CSV not found: {str(csv_path)}")

        concept_names = tuple(concept_specs.keys())
        label_to_index_by_concept = {
            concept: {str(label).lower(): int(i) for i, label in enumerate(labels)}
            for concept, labels in concept_specs.items()
        }

        exam_id_to_targets: Dict[str, Tuple[int, ...]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                echo_id = row.get("echo_id", None)
                if echo_id is None:
                    continue
                exam_id = str(echo_id).strip()
                if not exam_id:
                    continue

                targets: List[int] = []
                for concept in concept_names:
                    normalized = _normalize_concept_value(row.get(concept, None))
                    if normalized is None:
                        targets.append(int(ignore_index))
                        continue

                    mapping = label_to_index_by_concept.get(concept, {})
                    idx = mapping.get(normalized, None)
                    if idx is None:
                        targets.append(int(ignore_index))
                        continue
                    targets.append(int(idx))

                exam_id_to_targets[exam_id] = tuple(targets)

        return cls(
            concept_names=concept_names,
            label_to_index_by_concept=label_to_index_by_concept,
            exam_id_to_targets=exam_id_to_targets,
            ignore_index=int(ignore_index),
        )

    def targets_for_exam_ids(self, exam_ids: Sequence[str]) -> torch.Tensor:
        num_concepts = len(self.concept_names)
        batch_targets = torch.full(
            (len(exam_ids), num_concepts),
            fill_value=int(self.ignore_index),
            dtype=torch.long,
        )
        for i, exam_id in enumerate(exam_ids):
            key = str(exam_id)
            targets = self.exam_id_to_targets.get(key, None)
            if targets is None:
                continue
            batch_targets[i, :] = torch.tensor(list(targets), dtype=torch.long)
        return batch_targets


def build_concept_and_label_special_tokens(concept_specs: Mapping[str, Sequence[str]]) -> List[str]:
    concept_tokens = [f"<{name}>" for name in concept_specs.keys()]
    label_tokens = sorted({f"<{label}>" for labels in concept_specs.values() for label in labels})
    return concept_tokens + label_tokens


def build_concept_token_ids(tokenizer: PreTrainedTokenizerBase, concept_names: Sequence[str]) -> torch.Tensor:
    tokens = [f"<{name}>" for name in concept_names]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(int(x) < 0 for x in ids):
        missing = [t for t, i in zip(tokens, ids) if int(i) < 0]
        raise ValueError(f"Some concept tokens are missing in tokenizer: {missing}")
    return torch.tensor([int(x) for x in ids], dtype=torch.long)


def build_concept_label_token_id_tensor(
    tokenizer: PreTrainedTokenizerBase,
    concept_specs: Mapping[str, Sequence[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    concept_names = list(concept_specs.keys())
    label_lists = [list(concept_specs[name]) for name in concept_names]
    max_labels = max((len(labels) for labels in label_lists), default=1)

    token_ids = torch.zeros((len(concept_names), max_labels), dtype=torch.long)
    token_mask = torch.zeros((len(concept_names), max_labels), dtype=torch.bool)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    for c_idx, labels in enumerate(label_lists):
        for l_idx, label in enumerate(labels):
            token = f"<{label}>"
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            if unk_id is not None and int(token_id) == int(unk_id):
                raise ValueError(f"Tokenizer returned unk for label token={token}")
            token_ids[c_idx, l_idx] = int(token_id)
            token_mask[c_idx, l_idx] = True

    return token_ids, token_mask


def enable_training_on_new_tokens(
    model: nn.Module,
    frozen_vocab_size: int,
) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return None

    weight = input_embeddings.weight
    weight.requires_grad = True

    frozen_rows = int(frozen_vocab_size)

    def grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        grad = grad.clone()
        grad[:frozen_rows].zero_()
        return grad

    weight.register_hook(grad_mask_hook)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        return None

    if hasattr(output_embeddings, "weight") and output_embeddings.weight is not weight:
        output_weight = output_embeddings.weight
        output_weight.requires_grad = True
        output_weight.register_hook(grad_mask_hook)

    return None


class ConceptAwareVlmDataCollator:
    def __init__(self, base_collator: Any, concept_table: ConceptLabelTable) -> None:
        self.base_collator = base_collator
        self.concept_table = concept_table

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(examples)
        exam_ids = [str(x) for x in list(batch.get("exam_id", []))]
        batch["concept_targets"] = self.concept_table.targets_for_exam_ids(exam_ids)
        return batch


def masked_cross_entropy_mean(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    per_example = f.cross_entropy(
        logits,
        targets,
        ignore_index=int(ignore_index),
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    valid = targets.ne(int(ignore_index))
    denom = valid.sum().clamp_min(1).to(dtype=per_example.dtype)
    return (per_example * valid.to(dtype=per_example.dtype)).sum() / denom


def compute_drw_blend_factor(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    """Compute a DRW (Deferred Re-Weighting) blend factor.

    A value of 0.0 corresponds to uniform (no re-weighting), and 1.0 corresponds
    to the full re-weighting scheme.

    Args:
        epoch: Current epoch index (0-based).
        start_epoch: Epoch index at which re-weighting begins.
        ramp_epochs: Number of epochs over which the blend factor is linearly
            ramped from 0 to 1. If <= 0, the factor jumps to 1 at start_epoch.

    Returns:
        Blend factor in [0.0, 1.0].
    """
    epoch_index = int(epoch)
    start_index = int(start_epoch)
    ramp = int(ramp_epochs)

    if epoch_index < start_index:
        return 0.0
    if ramp <= 0:
        return 1.0

    progress = float(epoch_index - start_index + 1) / float(ramp)
    return float(max(0.0, min(1.0, progress)))


def compute_concept_class_counts(
    exam_ids: Sequence[str],
    concept_table: "ConceptLabelTable",
    concept_specs: Mapping[str, Sequence[str]],
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    counts_by_concept: Dict[str, torch.Tensor] = {}
    if not exam_ids:
        for concept_name, labels in concept_specs.items():
            counts_by_concept[str(concept_name)] = torch.zeros(
                (int(len(labels)),), dtype=torch.long
            )
        return counts_by_concept

    targets = concept_table.targets_for_exam_ids(exam_ids)

    for concept_index, (concept_name, labels) in enumerate(concept_specs.items()):
        num_labels = int(len(labels))
        if num_labels <= 0:
            counts_by_concept[str(concept_name)] = torch.zeros((0,), dtype=torch.long)
            continue

        concept_targets = targets[:, int(concept_index)]
        valid = concept_targets.ne(int(ignore_index))
        valid &= concept_targets.ge(0)
        valid &= concept_targets.lt(int(num_labels))
        valid_targets = concept_targets[valid]

        if int(valid_targets.numel()) == 0:
            counts_by_concept[str(concept_name)] = torch.zeros(
                (num_labels,), dtype=torch.long
            )
            continue

        bincount = torch.bincount(valid_targets, minlength=num_labels)[:num_labels]
        counts_by_concept[str(concept_name)] = bincount.to(dtype=torch.long)

    return counts_by_concept


def _normalize_positive_weights(weights: torch.Tensor, eps: float) -> torch.Tensor:
    positive = weights.gt(0)
    if not bool(positive.any().item()):
        return torch.ones_like(weights)

    mean_value = weights[positive].mean().clamp_min(float(eps))
    return weights / mean_value


def compute_class_balanced_weights_from_counts(
    counts: torch.Tensor,
    beta: float,
    max_weight: float,
    weight_power: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    beta_value = float(beta)
    eps_value = float(eps)

    counts_float = counts.to(dtype=torch.float64)
    effective_num = 1.0 - torch.pow(torch.tensor(beta_value, dtype=torch.float64), counts_float)
    weights = (1.0 - beta_value) / (effective_num + eps_value)
    weights = weights.to(dtype=torch.float32)

    weights = torch.where(counts.gt(0), weights, torch.zeros_like(weights))
    weights = _normalize_positive_weights(weights, eps=eps_value)

    power = float(weight_power)
    if power != 1.0:
        weights = torch.pow(weights.clamp_min(eps_value), power)

    max_w = float(max_weight)
    if max_w > 0:
        weights = weights.clamp(max=max_w)

    return weights


def compute_inverse_frequency_weights_from_counts(
    counts: torch.Tensor,
    max_weight: float,
    weight_power: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    eps_value = float(eps)
    counts_float = counts.to(dtype=torch.float32).clamp_min(0)

    weights = 1.0 / (counts_float + eps_value)
    weights = torch.where(counts.gt(0), weights, torch.zeros_like(weights))
    weights = _normalize_positive_weights(weights, eps=eps_value)

    power = float(weight_power)
    if power != 1.0:
        weights = torch.pow(weights.clamp_min(eps_value), power)

    max_w = float(max_weight)
    if max_w > 0:
        weights = weights.clamp(max=max_w)

    return weights


def build_concept_class_weight_tensor(
    concept_specs: Mapping[str, Sequence[str]],
    class_counts_by_concept: Mapping[str, torch.Tensor],
    strategy: str,
    cb_beta: float,
    weight_power: float,
    max_weight: float,
) -> torch.Tensor:
    strategy_name = str(strategy).strip().lower()
    concept_names = list(concept_specs.keys())
    label_lists = [list(concept_specs[name]) for name in concept_names]
    max_labels = max((len(labels) for labels in label_lists), default=1)

    weight_tensor = torch.ones((len(concept_names), int(max_labels)), dtype=torch.float32)
    if strategy_name in {"", "none"}:
        return weight_tensor

    for concept_index, concept_name in enumerate(concept_names):
        labels = list(concept_specs.get(concept_name, []))
        num_labels = int(len(labels))
        if num_labels <= 0:
            continue

        counts = class_counts_by_concept.get(str(concept_name), None)
        if counts is None:
            counts = torch.zeros((num_labels,), dtype=torch.long)

        counts = counts.to(dtype=torch.long).reshape(-1)[:num_labels]

        if strategy_name in {"inverse_freq", "weighted_ce"}:
            weights = compute_inverse_frequency_weights_from_counts(
                counts=counts,
                max_weight=float(max_weight),
                weight_power=float(weight_power),
            )
        else:
            weights = compute_class_balanced_weights_from_counts(
                counts=counts,
                beta=float(cb_beta),
                max_weight=float(max_weight),
                weight_power=float(weight_power),
            )

        weight_tensor[int(concept_index), :num_labels] = weights.to(dtype=torch.float32)

    return weight_tensor


def blend_concept_class_weights(
    full_weights: Optional[torch.Tensor],
    blend_factor: float,
) -> Optional[torch.Tensor]:
    if full_weights is None:
        return None

    alpha = float(max(0.0, min(1.0, float(blend_factor))))
    if alpha <= 0.0:
        return torch.ones_like(full_weights)
    if alpha >= 1.0:
        return full_weights

    return (1.0 - alpha) * torch.ones_like(full_weights) + alpha * full_weights

def concept_ce_sliced_per_concept(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_labels_by_concept: Sequence[int],
    ignore_index: int,
    class_weight_by_concept: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits_fp32 = logits.float()
    loss_values = []
    num_concepts = int(targets.shape[1])

    for concept_index in range(num_concepts):
        num_labels = int(num_labels_by_concept[concept_index])
        if num_labels <= 1:
            continue

        concept_targets = targets[:, concept_index]
        valid = concept_targets.ne(int(ignore_index))
        valid &= concept_targets.ge(0)
        valid &= concept_targets.lt(num_labels)

        if not bool(valid.any().item()):
            continue

        concept_logits = logits_fp32[:, concept_index, :num_labels]
        per_example = f.cross_entropy(
            concept_logits[valid],
            concept_targets[valid],
            reduction="none",
            label_smoothing=float(label_smoothing),
        )

        if class_weight_by_concept is not None:
            weights = class_weight_by_concept.to(device=targets.device)[
                concept_index, concept_targets[valid]
            ].to(dtype=per_example.dtype)
            per_example = per_example * weights

        loss_values.append(per_example.mean())

    if not loss_values:
        return torch.zeros((), device=targets.device, dtype=torch.float32)

    return torch.stack(loss_values).mean()



def masked_weighted_cross_entropy_mean_by_concept(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight_by_concept: Optional[torch.Tensor],
    ignore_index: int,
    label_smoothing: float = 0.0,
    balance_across_concepts: bool = False,
) -> torch.Tensor:
    if logits is None:
        return torch.zeros((), device=targets.device, dtype=torch.float32)

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (B, C, K), got {tuple(logits.shape)}")

    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (B, C), got {tuple(targets.shape)}")

    batch_size, num_concepts, num_classes = logits.shape

    if targets.shape[0] != batch_size or targets.shape[1] != num_concepts:
        raise ValueError(
            "targets shape must match logits[:2]. "
            f"logits[:2]={tuple(logits.shape[:2])} targets={tuple(targets.shape)}"
        )

    label_smoothing_value = float(max(0.0, min(0.999, float(label_smoothing))))
    device = logits.device
    device_type = device.type

    with torch.autocast(device_type=device_type, enabled=False):
        logits_fp32 = logits.float()
        targets_long = targets.to(device=device, dtype=torch.long)

        flat_logits = logits_fp32.reshape(-1, int(num_classes))
        flat_targets = targets_long.reshape(-1)

        per_element_loss = f.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=int(ignore_index),
            reduction="none",
            label_smoothing=label_smoothing_value,
        )

        valid = flat_targets.ne(int(ignore_index))
        valid &= flat_targets.ge(0)
        valid &= flat_targets.lt(int(num_classes))

        if class_weight_by_concept is not None and bool(valid.any().item()):
            weights_fp32 = class_weight_by_concept.to(device=device, dtype=torch.float32)

            if weights_fp32.ndim != 2:
                raise ValueError(
                    f"class_weight_by_concept must have shape (C, K), got {tuple(weights_fp32.shape)}"
                )
            if weights_fp32.shape[0] != num_concepts or weights_fp32.shape[1] != num_classes:
                raise ValueError(
                    "class_weight_by_concept shape must match (num_concepts, num_classes). "
                    f"Expected ({num_concepts}, {num_classes}), got {tuple(weights_fp32.shape)}"
                )

            concept_index_matrix = torch.arange(
                int(num_concepts),
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(batch_size), -1)

            flat_concept_indices = concept_index_matrix.reshape(-1)
            selected_weights = weights_fp32[flat_concept_indices[valid], flat_targets[valid]]

            per_element_loss = per_element_loss.clone()
            per_element_loss[valid] = per_element_loss[valid] * selected_weights

        if not bool(balance_across_concepts):
            valid_fp32 = valid.to(dtype=per_element_loss.dtype)
            denom = valid_fp32.sum().clamp_min(1.0)
            return (per_element_loss * valid_fp32).sum() / denom

        loss_matrix = per_element_loss.reshape(int(batch_size), int(num_concepts))
        valid_matrix = valid.reshape(int(batch_size), int(num_concepts))

        valid_counts = valid_matrix.sum(dim=0).to(dtype=torch.float32)
        concept_has_valid = valid_counts.gt(0)

        if not bool(concept_has_valid.any().item()):
            return torch.zeros((), device=device, dtype=torch.float32)

        loss_sums = (loss_matrix * valid_matrix.to(dtype=loss_matrix.dtype)).sum(dim=0)
        concept_means = loss_sums / valid_counts.clamp_min(1.0).to(dtype=loss_sums.dtype)

        return concept_means[concept_has_valid].mean()



def _extract_exam_id_from_item(item: Any) -> Optional[str]:
    if isinstance(item, Mapping):
        for key in ("exam_id", "echo_id", "study_id", "id"):
            value = item.get(key, None)
            if value is not None and str(value).strip():
                return str(value).strip()

    for attr in ("exam_id", "echo_id", "study_id", "id"):
        if hasattr(item, attr):
            value = getattr(item, attr)
            if value is not None and str(value).strip():
                return str(value).strip()

    return None


def extract_exam_ids_from_dataset(dataset: Any) -> List[str]:
    indices: Optional[Sequence[int]] = None
    base_dataset = dataset

    if isinstance(dataset, Subset):
        indices = list(dataset.indices)
        base_dataset = dataset.dataset

    for attr_name in ("items", "data", "examples"):
        if hasattr(base_dataset, attr_name):
            candidate = getattr(base_dataset, attr_name)
            if isinstance(candidate, Sequence):
                items = candidate
                if indices is None:
                    indices = range(len(items))
                exam_ids = []
                for i in indices:
                    exam_id = _extract_exam_id_from_item(items[int(i)])
                    if exam_id is not None:
                        exam_ids.append(exam_id)
                if int(len(exam_ids)) == int(len(list(indices))):
                    return exam_ids

    if indices is None:
        indices = range(len(base_dataset))

    exam_ids: List[str] = []
    for i in indices:
        example = base_dataset[int(i)]
        exam_id = _extract_exam_id_from_item(example)
        if exam_id is None:
            raise ValueError(
                "Could not extract exam_id/echo_id from dataset examples."
            )
        exam_ids.append(exam_id)

    return exam_ids


def compute_sample_weights_for_concept_balanced_sampler(
    exam_ids: Sequence[str],
    concept_table: "ConceptLabelTable",
    class_weight_by_concept: torch.Tensor,
    ignore_index: int,
    reduction: str = "max",
    weight_power: float = 1.0,
    max_weight: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not exam_ids:
        return torch.ones((0,), dtype=torch.double)

    targets = concept_table.targets_for_exam_ids(exam_ids)
    num_samples, num_concepts = targets.shape

    concept_indices = torch.arange(int(num_concepts), dtype=torch.long).unsqueeze(0)
    concept_indices = concept_indices.expand(int(num_samples), -1)

    valid = targets.ne(int(ignore_index))
    valid &= targets.ge(0)
    valid &= targets.lt(int(class_weight_by_concept.shape[1]))

    weights_per_element = torch.zeros(
        (int(num_samples), int(num_concepts)), dtype=torch.float32
    )

    if bool(valid.any().item()):
        selected = class_weight_by_concept[concept_indices[valid], targets[valid]]
        weights_per_element[valid] = selected.to(dtype=torch.float32)

    reduction_name = str(reduction).strip().lower()
    if reduction_name == "mean":
        denom = valid.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
        sample_weights = weights_per_element.sum(dim=1) / denom
    else:
        sample_weights = weights_per_element.max(dim=1).values
        sample_weights = torch.where(
            sample_weights.gt(0), sample_weights, torch.ones_like(sample_weights)
        )

    sample_weights = _normalize_positive_weights(
        sample_weights.to(dtype=torch.float32), eps=float(eps)
    )

    power = float(weight_power)
    if power != 1.0:
        sample_weights = torch.pow(sample_weights.clamp_min(float(eps)), power)

    max_w = float(max_weight)
    if max_w > 0:
        sample_weights = sample_weights.clamp(max=max_w)

    return sample_weights.to(dtype=torch.double)



def _concept_label_from_index(
    concept_name: str,
    label_index: int,
    concept_specs: Mapping[str, Sequence[str]],
) -> str:
    labels = list(concept_specs.get(str(concept_name), []))
    if int(label_index) < 0 or int(label_index) >= int(len(labels)):
        return ""
    return str(labels[int(label_index)])


def _balanced_accuracy_multiclass(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> float:
    if int(targets.size) == 0:
        return 0.0

    recalls: List[float] = []
    for class_index in range(int(num_classes)):
        support = int(np.sum(targets == int(class_index)))
        if support == 0:
            continue

        true_pos = int(
            np.sum((targets == int(class_index)) & (predictions == int(class_index)))
        )
        recalls.append(float(true_pos) / float(support))

    if not recalls:
        return 0.0
    return float(np.mean(np.asarray(recalls, dtype=np.float64)))


def _macro_f1_multiclass(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> float:
    if int(targets.size) == 0:
        return 0.0

    f1_values: List[float] = []
    for class_index in range(int(num_classes)):
        support = int(np.sum(targets == int(class_index)))
        if support == 0:
            continue

        true_pos = int(
            np.sum((targets == int(class_index)) & (predictions == int(class_index)))
        )
        false_pos = int(
            np.sum((targets != int(class_index)) & (predictions == int(class_index)))
        )
        false_neg = int(
            np.sum((targets == int(class_index)) & (predictions != int(class_index)))
        )

        denom = 2 * true_pos + false_pos + false_neg
        value = float(2 * true_pos) / float(denom) if denom > 0 else 0.0
        f1_values.append(value)

    if not f1_values:
        return 0.0
    return float(np.mean(np.asarray(f1_values, dtype=np.float64)))


@torch.no_grad()
def run_test_generation_and_save_metrics(
    model: EchoReportVlm,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder: ReportPromptBuilder,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_prompt_tokens: int,
    gen_max_new_tokens: int,
    metrics_computer: ReportMetricsComputer,
    output_csv_path: Path,
    concept_names: Sequence[str],
    concept_specs: Mapping[str, Sequence[str]],
    ignore_index: int,
) -> Dict[str, float]:
    model.eval()

    exam_ids: List[str] = []
    gt_reports: List[str] = []
    generated_reports: List[str] = []
    reasoning_traces: List[str] = []

    concept_gt_indices_by_example: List[List[int]] = []
    concept_pred_indices_by_example: List[List[int]] = []

    for batch in loader:
        batch_exam_ids = list(batch["exam_id"])
        batch_gt_reports = list(batch["gt_report"])
        masked_reports = list(batch.get("masked_report", [""] * len(batch_exam_ids)))

        video_features = batch["video_features"].to(
            device=device, dtype=autocast_dtype, non_blocking=True
        )
        video_mask = batch["video_mask"].to(device=device, non_blocking=True)

        prompts = [prompt_builder.build(masked_report=m) for m in masked_reports]
        prompt_ids = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=int(max_prompt_tokens),
            padding=True,
            return_tensors="pt",
        )
        prompt_input_ids = prompt_ids["input_ids"].to(device=device, non_blocking=True)
        prompt_attention_mask = prompt_ids["attention_mask"].to(
            device=device, non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(device.type == "cuda"),
        ):
            batch_generated_texts = model.generate_report(
                tokenizer=tokenizer,
                video_features=video_features,
                video_mask=video_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                max_new_tokens=int(gen_max_new_tokens),
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                study_ids=batch_exam_ids if batch_exam_ids else None,
            )

        batch_concept_targets = batch.get("concept_targets", None)
        if batch_concept_targets is not None:
            concept_targets = batch_concept_targets.to(device=device, non_blocking=True)
            input_ids = batch["input_ids"].to(device=device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device=device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=(device.type == "cuda"),
            ):
                pred_out = model(
                    video_features=video_features,
                    video_mask=video_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    concept_targets=concept_targets,
                    study_ids=batch.get("exam_id"),
                )

            concept_logits = getattr(pred_out, "concept_logits", None)
            function_logits = getattr(pred_out, "function_logits", None)
            logits_for_pred = concept_logits
            if concept_logits is not None and function_logits is not None:
                logits_for_pred = 0.5 * concept_logits + 0.5 * function_logits

            if logits_for_pred is not None:
                pred_indices = (
                    logits_for_pred.argmax(dim=-1).detach().cpu().to(dtype=torch.long)
                )
                concept_gt_indices_by_example.extend(
                    concept_targets.detach().cpu().to(dtype=torch.long).tolist()
                )
                concept_pred_indices_by_example.extend(pred_indices.tolist())

        for exam_id, gt_report, gen_text in zip(
            batch_exam_ids, batch_gt_reports, batch_generated_texts
        ):
            exam_ids.append(str(exam_id))
            gt_reports.append(str(gt_report))
            reasoning, report = parse_reasoning_and_report_from_generation(gen_text)
            generated_reports.append(report)
            reasoning_traces.append(reasoning)

    metrics = metrics_computer.compute(references=gt_reports, predictions=generated_reports)

    fieldnames = [
        "echo_id",
        "generated_report",
        "reasoning",
        "gt_report",
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "rouge_l",
        "meteor",
        "cider",
        "ce_precision",
        "ce_recall",
        "ce_f1",
    ]

    for concept_name in concept_names:
        fieldnames.append(f"{concept_name}_gt")
        fieldnames.append(f"{concept_name}_pred")
        fieldnames.append(f"{concept_name}_correct")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i, echo_id in enumerate(exam_ids):
            row: Dict[str, Any] = {
                "echo_id": str(echo_id),
                "generated_report": str(generated_reports[i]),
                "reasoning": str(reasoning_traces[i]),
                "gt_report": str(gt_reports[i]),
                "bleu_1": float(metrics.bleu_1[i]),
                "bleu_2": float(metrics.bleu_2[i]),
                "bleu_3": float(metrics.bleu_3[i]),
                "bleu_4": float(metrics.bleu_4[i]),
                "rouge_l": float(metrics.rouge_l[i]),
                "meteor": float(metrics.meteor[i]),
                "cider": float(metrics.cider[i]),
                #"bleurt": float(metrics.bleurt[i]),
                "ce_precision": float(metrics.ce_precision[i]),
                "ce_recall": float(metrics.ce_recall[i]),
                "ce_f1": float(metrics.ce_f1[i]),
            }

            if i < int(len(concept_gt_indices_by_example)) and i < int(
                len(concept_pred_indices_by_example)
            ):
                gt_indices = concept_gt_indices_by_example[i]
                pred_indices = concept_pred_indices_by_example[i]
                for concept_index, concept_name in enumerate(concept_names):
                    gt_index = int(gt_indices[int(concept_index)])
                    pred_index = int(pred_indices[int(concept_index)])
                    gt_label = "" if gt_index == int(ignore_index) else _concept_label_from_index(
                        concept_name=concept_name,
                        label_index=gt_index,
                        concept_specs=concept_specs,
                    )
                    pred_label = _concept_label_from_index(
                        concept_name=concept_name,
                        label_index=pred_index,
                        concept_specs=concept_specs,
                    )
                    correct_value: Any = ""
                    if gt_index != int(ignore_index):
                        correct_value = int(gt_index == pred_index)

                    row[f"{concept_name}_gt"] = gt_label
                    row[f"{concept_name}_pred"] = pred_label
                    row[f"{concept_name}_correct"] = correct_value

            writer.writerow(row)

    summary = {
        "bleu_1": _mean(metrics.bleu_1),
        "bleu_2": _mean(metrics.bleu_2),
        "bleu_3": _mean(metrics.bleu_3),
        "bleu_4": _mean(metrics.bleu_4),
        "rouge_l": _mean(metrics.rouge_l),
        "meteor": _mean(metrics.meteor),
        "cider": _mean(metrics.cider),
        #"bleurt": _mean(metrics.bleurt),
        "ce_precision": _mean(metrics.ce_precision),
        "ce_recall": _mean(metrics.ce_recall),
        "ce_f1": _mean(metrics.ce_f1),
        "num_examples": float(len(exam_ids)),
    }

    concept_balanced_acc_by_name: Dict[str, float] = {}
    concept_f1_by_name: Dict[str, float] = {}
    concept_weights: Dict[str, int] = {}

    if concept_gt_indices_by_example and concept_pred_indices_by_example:
        num_examples_with_concepts = min(
            int(len(concept_gt_indices_by_example)),
            int(len(concept_pred_indices_by_example)),
        )
        gt_matrix = np.asarray(
            concept_gt_indices_by_example[:num_examples_with_concepts],
            dtype=np.int64,
        )
        pred_matrix = np.asarray(
            concept_pred_indices_by_example[:num_examples_with_concepts],
            dtype=np.int64,
        )
        num_concepts_available = min(
            int(gt_matrix.shape[1]) if int(gt_matrix.ndim) == 2 else 0,
            int(pred_matrix.shape[1]) if int(pred_matrix.ndim) == 2 else 0,
            int(len(concept_names)),
        )

        for concept_index, concept_name in enumerate(concept_names):
            balanced_acc = 0.0
            f1_value = 0.0
            valid_count = 0

            class_names = list(concept_specs.get(str(concept_name), []))
            num_classes = int(len(class_names))

            if int(concept_index) < int(num_concepts_available) and num_classes > 1:
                targets = gt_matrix[:, int(concept_index)]
                predictions = pred_matrix[:, int(concept_index)]

                valid_mask = targets != int(ignore_index)
                valid_mask &= targets >= 0
                valid_mask &= targets < int(num_classes)

                valid_targets = targets[valid_mask]
                valid_predictions = predictions[valid_mask]
                valid_count = int(valid_targets.shape[0])

                balanced_acc = _balanced_accuracy_multiclass(
                    targets=valid_targets,
                    predictions=valid_predictions,
                    num_classes=num_classes,
                )
                f1_value = _macro_f1_multiclass(
                    targets=valid_targets,
                    predictions=valid_predictions,
                    num_classes=num_classes,
                )

            concept_key = str(concept_name)
            concept_balanced_acc_by_name[concept_key] = float(balanced_acc)
            concept_f1_by_name[concept_key] = float(f1_value)
            concept_weights[concept_key] = int(valid_count)

            summary[f"concept_balanced_acc/{concept_name}"] = float(balanced_acc)
            summary[f"concept_f1/{concept_name}"] = float(f1_value)
    else:
        for concept_name in concept_names:
            concept_key = str(concept_name)
            concept_balanced_acc_by_name[concept_key] = 0.0
            concept_f1_by_name[concept_key] = 0.0
            concept_weights[concept_key] = 0

            summary[f"concept_balanced_acc/{concept_name}"] = 0.0
            summary[f"concept_f1/{concept_name}"] = 0.0

    weight_total = int(sum(concept_weights.values()))
    if weight_total > 0:
        weighted_balanced_acc = sum(
            float(concept_balanced_acc_by_name[k]) * float(concept_weights[k])
            for k in concept_balanced_acc_by_name
        ) / float(weight_total)
        weighted_f1 = sum(
            float(concept_f1_by_name[k]) * float(concept_weights[k])
            for k in concept_f1_by_name
        ) / float(weight_total)
    else:
        weighted_balanced_acc = 0.0
        weighted_f1 = 0.0

    summary["concept_balanced_acc_overall"] = float(weighted_balanced_acc)
    summary["concept_f1_overall"] = float(weighted_f1)

    model.train()
    return summary



def compute_contrastive_losses(
    video_repr: torch.Tensor,
    text_repr: torch.Tensor,
    temperature: float,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(video_repr.shape[0])
    targets = torch.arange(batch_size, device=video_repr.device)

    sim = torch.matmul(video_repr, text_repr.transpose(0, 1)) / float(max(1e-8, temperature))
    loss_i2t = f.cross_entropy(sim, targets)
    loss_t2i = f.cross_entropy(sim.transpose(0, 1), targets)
    infonce_loss = 0.5 * (loss_i2t + loss_t2i)

    shuffled = torch.randperm(batch_size, device=video_repr.device)
    sim_pos = (video_repr * text_repr).sum(dim=1)
    sim_neg = (video_repr[shuffled] * text_repr).sum(dim=1)
    margin_loss = f.relu(float(margin) + sim_neg - sim_pos).mean()

    return infonce_loss, margin_loss


@dataclass(frozen=True)
class EvalLossWeights:
    generation_loss_weight: float
    concept_loss_weight: float
    contrastive_loss_weight: float
    contrastive_margin_weight: float
    contrastive_temperature: float
    contrastive_margin: float


def _safe_binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")

    try:
        return float(roc_auc_score(targets, scores))
    except Exception:
        return float("nan")


def _macro_one_vs_rest_auroc(
    multiclass_targets: np.ndarray,
    multiclass_probabilities: np.ndarray,
    class_names: Sequence[str],
    ignored_class_names: Sequence[str] = ("not_well_visualized", "indeterminate"),
) -> float:
    ignored = {str(x).lower() for x in ignored_class_names}
    auc_values: List[float] = []

    for class_index, class_name in enumerate(class_names):
        if str(class_name).lower() in ignored:
            continue

        binary_targets = (multiclass_targets == int(class_index)).astype(np.int32)
        positives = int(binary_targets.sum())
        if positives == 0 or positives == int(binary_targets.shape[0]):
            continue

        scores = multiclass_probabilities[:, int(class_index)]
        auc = _safe_binary_auroc(targets=binary_targets, scores=scores)
        if not float(np.isnan(auc)):
            auc_values.append(float(auc))

    if not auc_values:
        return float("nan")
    return float(np.mean(np.asarray(auc_values, dtype=np.float64)))


def run_eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    ignore_index: int,
    concept_names: Sequence[str],
    concept_specs: Mapping[str, Sequence[str]],
    region_to_indices: Mapping[str, Sequence[int]],
    loss_weights: EvalLossWeights,
) -> Dict[str, float]:
    model.eval()
    total_losses: List[float] = []
    gen_losses: List[float] = []
    concept_losses: List[float] = []
    contrastive_losses: List[float] = []
    contrastive_margin_losses: List[float] = []
    token_correct = 0
    token_total = 0

    correct_by_concept = torch.zeros((len(concept_names),), dtype=torch.long)
    total_by_concept = torch.zeros((len(concept_names),), dtype=torch.long)

    probability_chunks_by_concept: List[List[torch.Tensor]] = [[] for _ in range(len(concept_names))]
    target_chunks_by_concept: List[List[torch.Tensor]] = [[] for _ in range(len(concept_names))]
    prediction_chunks_by_concept: List[List[torch.Tensor]] = [[] for _ in range(len(concept_names))]

    with torch.no_grad():
        for batch in loader:
            video_features = batch["video_features"].to(device=device, dtype=autocast_dtype, non_blocking=True)
            video_mask = batch["video_mask"].to(device=device, non_blocking=True)
            input_ids = batch["input_ids"].to(device=device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device=device, non_blocking=True)
            labels = batch["labels"].to(device=device, non_blocking=True)
            concept_targets = batch["concept_targets"].to(device=device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    video_features=video_features,
                    video_mask=video_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    concept_targets=concept_targets,
                    study_ids=batch.get("exam_id"),
                )

            gen_loss = out.loss
            concept_loss = torch.zeros((), device=gen_loss.device, dtype=gen_loss.dtype)

            token_correct += int(out.token_correct.detach().cpu().item())
            token_total += int(out.token_total.detach().cpu().item())

            concept_logits = getattr(out, "concept_logits", None)
            function_logits = getattr(out, "function_logits", None)

            logits_for_metrics = concept_logits
            if concept_logits is not None and function_logits is not None:
                logits_for_metrics = 0.5 * concept_logits + 0.5 * function_logits

            if concept_logits is not None:
                flat_targets = concept_targets.reshape(-1)

                concept_ce = masked_cross_entropy_mean(
                    logits=concept_logits.reshape(-1, concept_logits.shape[-1]),
                    targets=flat_targets,
                    ignore_index=int(ignore_index),
                )

                if function_logits is not None:
                    function_ce = masked_cross_entropy_mean(
                        logits=function_logits.reshape(-1, function_logits.shape[-1]),
                        targets=flat_targets,
                        ignore_index=int(ignore_index),
                    )
                    concept_loss = 0.5 * concept_ce + 0.5 * function_ce
                else:
                    concept_loss = concept_ce

            if logits_for_metrics is not None:
                pred = logits_for_metrics.argmax(dim=-1)
                valid = concept_targets.ne(int(ignore_index))
                correct = pred.eq(concept_targets) & valid

                correct_by_concept += correct.sum(dim=0).detach().cpu().to(dtype=torch.long)
                total_by_concept += valid.sum(dim=0).detach().cpu().to(dtype=torch.long)

                logits_cpu = logits_for_metrics.detach().float().cpu()
                targets_cpu = concept_targets.detach().cpu()
                prob_cpu = f.softmax(logits_cpu, dim=-1)
                pred_cpu = pred.detach().cpu()

                for concept_index, concept_name in enumerate(concept_names):
                    class_names = list(concept_specs.get(concept_name, []))
                    num_classes = int(len(class_names))
                    if num_classes <= 1:
                        continue

                    valid_mask = targets_cpu[:, int(concept_index)].ne(int(ignore_index))
                    if not bool(valid_mask.any().item()):
                        continue

                    probs_slice = prob_cpu[valid_mask, int(concept_index), :num_classes]
                    targets_slice = targets_cpu[valid_mask, int(concept_index)]
                    preds_slice = pred_cpu[valid_mask, int(concept_index)]

                    probability_chunks_by_concept[int(concept_index)].append(probs_slice)
                    target_chunks_by_concept[int(concept_index)].append(targets_slice)
                    prediction_chunks_by_concept[int(concept_index)].append(preds_slice)

            contrastive_loss = torch.zeros((), device=gen_loss.device, dtype=gen_loss.dtype)
            margin_loss = torch.zeros((), device=gen_loss.device, dtype=gen_loss.dtype)
            if float(loss_weights.contrastive_loss_weight) > 0.0:
                video_repr = f.normalize(out.video_repr, dim=-1)
                text_repr = f.normalize(out.text_repr, dim=-1)
                contrastive_loss, margin_loss = compute_contrastive_losses(
                    video_repr=video_repr,
                    text_repr=text_repr,
                    temperature=float(loss_weights.contrastive_temperature),
                    margin=float(loss_weights.contrastive_margin),
                )

            total_loss = (
                float(loss_weights.generation_loss_weight) * gen_loss
                + float(loss_weights.concept_loss_weight) * concept_loss
                + float(loss_weights.contrastive_loss_weight) * contrastive_loss
                + float(loss_weights.contrastive_margin_weight) * margin_loss
            )

            total_losses.append(float(total_loss.detach().cpu().item()))
            gen_losses.append(float(gen_loss.detach().cpu().item()))
            concept_losses.append(float(concept_loss.detach().cpu().item()))
            contrastive_losses.append(float(contrastive_loss.detach().cpu().item()))
            contrastive_margin_losses.append(float(margin_loss.detach().cpu().item()))

    model.train()

    mean_total_loss = float(sum(total_losses) / max(1, len(total_losses)))
    mean_gen_loss = float(sum(gen_losses) / max(1, len(gen_losses)))
    mean_concept_loss = float(sum(concept_losses) / max(1, len(concept_losses)))
    mean_contrastive_loss = float(sum(contrastive_losses) / max(1, len(contrastive_losses)))
    mean_margin_loss = float(sum(contrastive_margin_losses) / max(1, len(contrastive_margin_losses)))

    token_acc = float(token_correct) / float(token_total) if token_total > 0 else 0.0

    overall_total = int(total_by_concept.sum().item())
    overall_correct = int(correct_by_concept.sum().item())
    concept_acc_overall = float(overall_correct) / float(overall_total) if overall_total > 0 else 0.0

    metrics: Dict[str, float] = {
        "loss": float(mean_total_loss),
        "gen_loss": float(mean_gen_loss),
        "token_acc": float(token_acc),
        "concept_loss": float(mean_concept_loss),
        "contrastive_loss": float(mean_contrastive_loss),
        "contrastive_margin_loss": float(mean_margin_loss),
        "concept_acc_overall": float(concept_acc_overall),
    }

    for i, name in enumerate(concept_names):
        denom = int(total_by_concept[i].item())
        num = int(correct_by_concept[i].item())
        metrics[f"concept_acc/{name}"] = float(num) / float(denom) if denom > 0 else 0.0
        metrics[f"concept_count/{name}"] = float(denom)

    for region, indices in region_to_indices.items():
        denom = int(total_by_concept[list(indices)].sum().item()) if indices else 0
        num = int(correct_by_concept[list(indices)].sum().item()) if indices else 0
        metrics[f"concept_acc_region/{region}"] = float(num) / float(denom) if denom > 0 else 0.0

    concept_balanced_acc_by_name: Dict[str, float] = {}
    concept_f1_by_name: Dict[str, float] = {}
    concept_weights: Dict[str, int] = {}

    for concept_index, concept_name in enumerate(concept_names):
        balanced_acc = 0.0
        f1_value = 0.0
        valid_count = 0

        class_names = list(concept_specs.get(str(concept_name), []))
        num_classes = int(len(class_names))

        target_chunks = target_chunks_by_concept[int(concept_index)]
        pred_chunks = prediction_chunks_by_concept[int(concept_index)]

        if num_classes > 1 and target_chunks and pred_chunks:
            targets = torch.cat(target_chunks, dim=0).numpy().astype(np.int64)
            predictions = torch.cat(pred_chunks, dim=0).numpy().astype(np.int64)

            valid_mask = targets != int(ignore_index)
            valid_mask &= targets >= 0
            valid_mask &= targets < int(num_classes)

            valid_targets = targets[valid_mask]
            valid_predictions = predictions[valid_mask]
            valid_count = int(valid_targets.shape[0])

            balanced_acc = _balanced_accuracy_multiclass(
                targets=valid_targets,
                predictions=valid_predictions,
                num_classes=num_classes,
            )
            f1_value = _macro_f1_multiclass(
                targets=valid_targets,
                predictions=valid_predictions,
                num_classes=num_classes,
            )

        concept_key = str(concept_name)
        concept_balanced_acc_by_name[concept_key] = float(balanced_acc)
        concept_f1_by_name[concept_key] = float(f1_value)
        concept_weights[concept_key] = int(valid_count)

        metrics[f"concept_balanced_acc/{concept_name}"] = float(balanced_acc)
        metrics[f"concept_f1/{concept_name}"] = float(f1_value)

    weight_total = int(sum(concept_weights.values()))
    if weight_total > 0:
        weighted_balanced_acc = sum(
            float(concept_balanced_acc_by_name[k]) * float(concept_weights[k])
            for k in concept_balanced_acc_by_name
        ) / float(weight_total)
        weighted_f1 = sum(
            float(concept_f1_by_name[k]) * float(concept_weights[k])
            for k in concept_f1_by_name
        ) / float(weight_total)
    else:
        weighted_balanced_acc = 0.0
        weighted_f1 = 0.0

    metrics["concept_balanced_acc_overall"] = float(weighted_balanced_acc)
    metrics["concept_f1_overall"] = float(weighted_f1)

    for concept_index, concept_name in enumerate(concept_names):
        prob_chunks = probability_chunks_by_concept[int(concept_index)]
        target_chunks = target_chunks_by_concept[int(concept_index)]
        if not prob_chunks or not target_chunks:
            metrics[f"concept_auroc/{concept_name}"] = float("nan")
            continue

        probabilities = torch.cat(prob_chunks, dim=0).numpy()
        targets = torch.cat(target_chunks, dim=0).numpy().astype(np.int64)
        class_names = list(concept_specs.get(concept_name, []))
        probabilities = probabilities[:, : len(class_names)]

        metrics[f"concept_auroc/{concept_name}"] = _macro_one_vs_rest_auroc(
            multiclass_targets=targets,
            multiclass_probabilities=probabilities,
            class_names=class_names,
        )

    return metrics


CHECKPOINT_METADATA_FILENAME = "checkpoint_meta.json"
LM_AUXILIARY_STATE_FILENAME = "lm_auxiliary_state.pt"
CHECKPOINT_RUNTIME_FIELDS: Tuple[str, ...] = (
    "num_visual_tokens",
    "projector_layers",
    "projector_hidden_ratio",
    "projector_dropout",
    "adapter_layers",
    "adapter_heads",
    "adapter_attn_dropout",
    "adapter_mlp_ratio",
    "adapter_mlp_dropout",
    "projected_feature_check",
    "projected_feature_cosine_threshold",
    "projected_feature_max_pairs_to_log",
    "lm_head_chunk_size",
)


def read_checkpoint_metadata(checkpoint_dir: Path) -> Dict[str, Any]:
    metadata_path = Path(checkpoint_dir) / CHECKPOINT_METADATA_FILENAME
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f_in:
        payload = json.load(f_in)

    if isinstance(payload, dict):
        return payload
    return {}


def resolve_base_model_name_or_path(
    checkpoint_dir: Path,
    provided_value: Optional[str],
) -> str:
    provided_text = "" if provided_value is None else str(provided_value).strip()
    if provided_text:
        return provided_text

    metadata = read_checkpoint_metadata(checkpoint_dir)
    stored_value = metadata.get("base_model_name_or_path", metadata.get("model_name_or_path", ""))
    stored_text = str(stored_value).strip()
    if stored_text:
        return stored_text

    raise ValueError(
        "base_model_name_or_path was not provided and was not found in checkpoint metadata."
    )


def _resolve_runtime_config_value(
    checkpoint_metadata: Mapping[str, Any],
    config: Any,
    field_name: str,
    default_value: Any,
) -> Any:
    if field_name in checkpoint_metadata:
        return checkpoint_metadata[field_name]
    if config is not None and hasattr(config, field_name):
        return getattr(config, field_name)
    return default_value


def _save_checkpoint_metadata(
    checkpoint_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[Any],
) -> None:
    metadata: Dict[str, Any] = {
        "checkpoint_format_version": 2,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "additional_special_tokens": list(
            dict.fromkeys(getattr(tokenizer, "additional_special_tokens", []))
        ),
    }

    if config is not None and hasattr(config, "model_name_or_path"):
        metadata["base_model_name_or_path"] = str(getattr(config, "model_name_or_path"))

    for field_name in CHECKPOINT_RUNTIME_FIELDS:
        if config is None or not hasattr(config, field_name):
            continue
        value = getattr(config, field_name)
        if isinstance(value, Path):
            value = str(value)
        metadata[field_name] = value

    metadata_path = Path(checkpoint_dir) / CHECKPOINT_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as f_out:
        json.dump(metadata, f_out, indent=2, sort_keys=True)


def _save_lm_auxiliary_state(
    checkpoint_dir: Path,
    lm_obj: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    input_embeddings = lm_obj.get_input_embeddings()
    output_embeddings = lm_obj.get_output_embeddings()

    input_weight = None
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        input_weight = input_embeddings.weight

    output_weight = None
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_weight = output_embeddings.weight

    special_tokens = list(
        dict.fromkeys(getattr(tokenizer, "additional_special_tokens", []))
    )
    tokens_to_save: List[str] = []
    token_ids_to_save: List[int] = []
    for token in special_tokens:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0:
            continue
        tokens_to_save.append(str(token))
        token_ids_to_save.append(int(token_id))

    auxiliary_state: Dict[str, Any] = {
        "schema_version": 1,
        "tokens": tokens_to_save,
    }

    if input_weight is not None and token_ids_to_save:
        input_token_ids = torch.tensor(
            token_ids_to_save,
            device=input_weight.device,
            dtype=torch.long,
        )
        auxiliary_state["input_embedding_rows"] = input_weight.detach().index_select(
            0,
            input_token_ids,
        ).cpu()

    if (
        output_weight is not None
        and token_ids_to_save
        and input_weight is not None
        and output_weight is not input_weight
    ):
        output_token_ids = torch.tensor(
            token_ids_to_save,
            device=output_weight.device,
            dtype=torch.long,
        )
        auxiliary_state["output_embedding_rows"] = output_weight.detach().index_select(
            0,
            output_token_ids,
        ).cpu()

    extra_trainable_parameters: Dict[str, torch.Tensor] = {}
    for name, param in lm_obj.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in str(name).lower():
            continue
        if input_weight is not None and param is input_weight:
            continue
        if output_weight is not None and param is output_weight:
            continue
        extra_trainable_parameters[str(name)] = param.detach().cpu()

    if extra_trainable_parameters:
        auxiliary_state["extra_trainable_parameters"] = extra_trainable_parameters

    should_save = bool(token_ids_to_save) or bool(extra_trainable_parameters)
    if not should_save:
        return None

    torch.save(
        auxiliary_state,
        Path(checkpoint_dir) / LM_AUXILIARY_STATE_FILENAME,
    )
    return None


def _load_lm_auxiliary_state(
    checkpoint_dir: Path,
    lm_obj: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    auxiliary_state_path = Path(checkpoint_dir) / LM_AUXILIARY_STATE_FILENAME
    if not auxiliary_state_path.exists():
        return None

    auxiliary_state = torch.load(auxiliary_state_path, map_location="cpu")
    if not isinstance(auxiliary_state, Mapping):
        raise ValueError(
            f"Invalid LM auxiliary state at {str(auxiliary_state_path)}."
        )

    tokens = [str(x) for x in list(auxiliary_state.get("tokens", []))]
    token_ids: List[int] = []
    for token in tokens:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0:
            raise ValueError(f"Tokenizer is missing checkpoint token: {token}")
        token_ids.append(int(token_id))

    input_embeddings = lm_obj.get_input_embeddings()
    output_embeddings = lm_obj.get_output_embeddings()

    input_weight = None
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        input_weight = input_embeddings.weight

    output_weight = None
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_weight = output_embeddings.weight

    input_rows = auxiliary_state.get("input_embedding_rows", None)
    if input_rows is not None:
        if input_weight is None:
            raise RuntimeError("Checkpoint contains input embedding rows but model has no input embeddings.")
        if int(input_rows.shape[0]) != int(len(token_ids)):
            raise ValueError("Input embedding row count does not match saved checkpoint tokens.")
        input_token_ids = torch.tensor(
            token_ids,
            device=input_weight.device,
            dtype=torch.long,
        )
        input_weight.data.index_copy_(
            0,
            input_token_ids,
            input_rows.to(device=input_weight.device, dtype=input_weight.dtype),
        )

    output_rows = auxiliary_state.get("output_embedding_rows", None)
    if output_rows is not None:
        if output_weight is None:
            raise RuntimeError("Checkpoint contains output embedding rows but model has no output embeddings.")
        if int(output_rows.shape[0]) != int(len(token_ids)):
            raise ValueError("Output embedding row count does not match saved checkpoint tokens.")
        output_token_ids = torch.tensor(
            token_ids,
            device=output_weight.device,
            dtype=torch.long,
        )
        output_weight.data.index_copy_(
            0,
            output_token_ids,
            output_rows.to(device=output_weight.device, dtype=output_weight.dtype),
        )

    extra_trainable_parameters = auxiliary_state.get("extra_trainable_parameters", None)
    if isinstance(extra_trainable_parameters, Mapping) and extra_trainable_parameters:
        lm_obj.load_state_dict(dict(extra_trainable_parameters), strict=False)

    return None


def _save_checkpoint_artifacts(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[Any] = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    base_model = unwrap_compiled_module(model)
    if not hasattr(base_model, "lm") or not hasattr(base_model, "adapter"):
        raise RuntimeError("Model must expose .lm and .adapter for checkpointing.")

    lm_obj = getattr(base_model, "lm")
    if isinstance(lm_obj, PeftModel):
        lm_obj.save_pretrained(checkpoint_dir / "lm_lora")
    else:
        raise RuntimeError("Checkpoint saver expects model.lm to be a PeftModel.")

    _save_lm_auxiliary_state(
        checkpoint_dir=checkpoint_dir,
        lm_obj=lm_obj,
        tokenizer=tokenizer,
    )

    adapter_obj = getattr(base_model, "adapter")
    torch.save(adapter_obj.state_dict(), checkpoint_dir / "adapter.pt")

    _save_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
        config=config,
    )


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[Any] = None,
) -> None:
    checkpoint_dir = Path(output_dir) / f"checkpoint-{int(step)}"
    _save_checkpoint_artifacts(
        checkpoint_dir=checkpoint_dir,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


def save_best_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    val_loss: float,
    val_report_acc: float,
    config: Optional[Any] = None,
) -> None:
    best_dir = Path(output_dir) / "checkpoint-best"
    if best_dir.exists():
        import shutil

        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    _save_checkpoint_artifacts(
        checkpoint_dir=best_dir,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    meta = {
        "step": int(step),
        "val_loss": float(val_loss),
        "val_report_word_acc": float(val_report_acc),
    }
    with (best_dir / "best_metrics.json").open("w", encoding="utf-8") as f_out:
        json.dump(meta, f_out, indent=2, sort_keys=True)


def _parse_checkpoint_step(dir_path: Path) -> Optional[int]:
    name = dir_path.name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    return None


def resolve_checkpoint_dir(checkpoint_root: Path) -> Path:
    checkpoint_root = Path(checkpoint_root)
    if (checkpoint_root / "tokenizer").exists():
        return checkpoint_root

    best_metrics_path = checkpoint_root / "best_metrics.json"
    if best_metrics_path.exists():
        with best_metrics_path.open("r", encoding="utf-8") as f_in:
            meta = json.load(f_in)
        step = meta.get("step", None)
        if step is not None:
            candidate = checkpoint_root / f"checkpoint-{int(step)}"
            if (candidate / "tokenizer").exists():
                return candidate

    candidates = [p for p in checkpoint_root.glob("checkpoint-*") if (p / "tokenizer").exists()]
    if candidates:
        candidates_sorted = sorted(
            candidates,
            key=lambda p: (_parse_checkpoint_step(p) is None, _parse_checkpoint_step(p) or -1),
        )
        return candidates_sorted[-1]

    raise FileNotFoundError(f"Could not find tokenizer under: {str(checkpoint_root)}")


def _load_qwen3_vl_for_training(
    model_name_or_path: str,
    from_pretrained_kwargs: Dict[str, Any],
) -> PreTrainedModel:
    model_path_lower = str(model_name_or_path).lower()
    load_attempts: List[Tuple[str, Any]] = []

    if "moe" in model_path_lower:
        load_attempts.append(("Qwen3VLMoeForConditionalGeneration", Qwen3VLMoeForConditionalGeneration))
    load_attempts.append(("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration))
    load_attempts.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
    load_attempts.append(("AutoModelForCausalLM", AutoModelForCausalLM))

    last_error: Optional[Exception] = None

    for _, cls in load_attempts:
        try:
            return cls.from_pretrained(model_name_or_path, **from_pretrained_kwargs)
        except TypeError:
            sanitized = dict(from_pretrained_kwargs)
            sanitized.pop("attn_implementation", None)
            try:
                return cls.from_pretrained(model_name_or_path, **sanitized)
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load model from {model_name_or_path}. Last error: {last_error}")


def load_best_model_from_checkpoint(
    checkpoint_dir: Path,
    base_model_name_or_path: Optional[str],
    config: Any,
    torch_dtype: torch.dtype,
    concept_label_token_ids: torch.Tensor,
    concept_label_token_mask: torch.Tensor,
    concept_token_ids: torch.Tensor,
) -> Tuple[EchoReportVlm, PreTrainedTokenizerBase]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_metadata = read_checkpoint_metadata(checkpoint_dir)
    resolved_base_model_name_or_path = resolve_base_model_name_or_path(
        checkpoint_dir=checkpoint_dir,
        provided_value=base_model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir / "tokenizer",
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_lm = _load_qwen3_vl_for_training(
        model_name_or_path=resolved_base_model_name_or_path,
        from_pretrained_kwargs={"torch_dtype": torch_dtype, "trust_remote_code": True},
    )
    base_lm.resize_token_embeddings(len(tokenizer))
    lm = PeftModel.from_pretrained(base_lm, checkpoint_dir / "lm_lora")

    auxiliary_state_path = checkpoint_dir / LM_AUXILIARY_STATE_FILENAME
    if auxiliary_state_path.exists():
        _load_lm_auxiliary_state(
            checkpoint_dir=checkpoint_dir,
            lm_obj=lm,
            tokenizer=tokenizer,
        )
    elif list(getattr(tokenizer, "additional_special_tokens", [])):
        print(
            "Warning: checkpoint is missing lm_auxiliary_state.pt. "
            "This usually means the checkpoint was saved before special-token embeddings "
            "and other non-LoRA LM weights were exported. Loading will be incomplete."
        )

    num_visual_tokens = int(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "num_visual_tokens",
            256,
        )
    )
    projector_layers = int(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "projector_layers",
            2,
        )
    )
    projector_hidden_ratio = float(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "projector_hidden_ratio",
            2.0,
        )
    )
    projector_dropout = float(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "projector_dropout",
            0.2,
        )
    )
    adapter_layers = int(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "adapter_layers",
            4,
        )
    )
    adapter_heads = int(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "adapter_heads",
            8,
        )
    )
    adapter_attn_dropout = float(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "adapter_attn_dropout",
            0.1,
        )
    )
    adapter_mlp_ratio = float(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "adapter_mlp_ratio",
            4.0,
        )
    )
    adapter_mlp_dropout = float(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "adapter_mlp_dropout",
            0.1,
        )
    )
    projected_feature_check = bool(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "projected_feature_check",
            True,
        )
    )
    projected_feature_cosine_threshold = float(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "projected_feature_cosine_threshold",
            0.9995,
        )
    )
    projected_feature_max_pairs_to_log = int(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "projected_feature_max_pairs_to_log",
            8,
        )
    )
    lm_head_chunk_size = int(
        _resolve_runtime_config_value(
            checkpoint_metadata,
            config,
            "lm_head_chunk_size",
            64,
        )
    )

    hidden_size = get_lm_hidden_size(lm)
    concept_query_init = lm.get_input_embeddings()(concept_token_ids).detach().to(dtype=torch.float32)

    adapter = VideoFeatureAdapter(
        input_dim=768,
        lm_hidden_size=hidden_size,
        num_report_tokens=num_visual_tokens,
        num_concept_tokens=int(len(CONCEPT_NAMES)),
        concept_query_init=concept_query_init,
        projector_layers=projector_layers,
        projector_hidden_ratio=projector_hidden_ratio,
        projector_dropout=projector_dropout,
        num_layers=adapter_layers,
        num_heads=adapter_heads,
        attn_dropout=adapter_attn_dropout,
        mlp_ratio=adapter_mlp_ratio,
        mlp_dropout=adapter_mlp_dropout,
        enable_projected_feature_check=projected_feature_check,
        projected_feature_cosine_threshold=projected_feature_cosine_threshold,
        projected_feature_max_pairs_to_log=projected_feature_max_pairs_to_log,
        concept_names=CONCEPT_NAMES,
    )
    adapter.to(dtype=torch_dtype)
    adapter_state = torch.load(checkpoint_dir / "adapter.pt", map_location="cpu")
    adapter.load_state_dict(adapter_state)

    model = EchoReportVlm(
        lm=lm,
        adapter=adapter,
        num_report_visual_tokens=num_visual_tokens,
        num_concept_tokens=int(len(CONCEPT_NAMES)),
        concept_label_token_ids=concept_label_token_ids,
        concept_label_token_mask=concept_label_token_mask,
        lm_head_chunk_size=lm_head_chunk_size,
    )
    return model, tokenizer


def is_no_decay_parameter(name: str, param: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    if name.endswith(".bias"):
        return True
    if param.ndim == 1:
        return True
    if "norm" in lowered:
        return True
    if "layernorm" in lowered:
        return True
    if "embed" in lowered:
        return True
    if "embedding" in lowered:
        return True
    if "lm_head" in lowered:
        return True
    return False


def _get_decay_parameter_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, param in model.named_parameters():
        if is_no_decay_parameter(name=name, param=param):
            continue
        names.append(name)
    return names


def build_qwen3_vl_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
    mm_projector_lr: Optional[float],
    vision_tower_lr: Optional[float],
) -> List[Dict[str, Any]]:
    decay_parameters = _get_decay_parameter_names(model)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    projector_parameters = [name for name, _ in model.named_parameters() if ("merger" in name or "adapter" in name)]
    vision_tower_parameters = [name for name, _ in model.named_parameters() if "visual" in name]

    grouped: List[Dict[str, Any]] = []

    if mm_projector_lr is not None and float(mm_projector_lr) != 0.0:
        if vision_tower_lr is not None and float(vision_tower_lr) != 0.0:
            grouped = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n in decay_parameters
                            and n not in projector_parameters
                            and n not in vision_tower_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": float(weight_decay),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n in decay_parameters
                            and n not in projector_parameters
                            and n in vision_tower_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": float(weight_decay),
                    "lr": float(vision_tower_lr),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n not in projector_parameters
                            and n not in vision_tower_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n not in projector_parameters
                            and n in vision_tower_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                    "lr": float(vision_tower_lr),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(weight_decay),
                    "lr": float(mm_projector_lr),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(mm_projector_lr),
                },
            ]
        else:
            grouped = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(weight_decay),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(weight_decay),
                    "lr": float(mm_projector_lr),
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(mm_projector_lr),
                },
            ]
    else:
        grouped = [
            {
                "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": float(weight_decay),
            },
            {
                "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]

    grouped = [g for g in grouped if g.get("params")]
    if not grouped:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return grouped


def create_qwen3_vl_finetuning_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    mm_projector_lr: Optional[float],
    vision_tower_lr: Optional[float],
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    use_fused: bool,
) -> torch.optim.Optimizer:
    param_groups = build_qwen3_vl_optimizer_param_groups(
        model=model,
        weight_decay=float(weight_decay),
        mm_projector_lr=mm_projector_lr,
        vision_tower_lr=vision_tower_lr,
    )

    kwargs: Dict[str, Any] = {
        "lr": float(base_lr),
        "betas": (float(adam_beta1), float(adam_beta2)),
        "eps": float(adam_eps),
    }

    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if use_fused and supports_fused and torch.cuda.is_available():
        kwargs["fused"] = True

    return torch.optim.AdamW(param_groups, **kwargs)


def create_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    scheduler_type: str,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(max(1, num_training_steps))
    warmup_steps = int(max(0, min(num_warmup_steps, total_steps - 1)))

    base_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
    min_lr_value = float(max(0.0, min_lr))
    min_lr_ratios: List[float] = []
    for base_lr in base_lrs:
        if base_lr <= 0.0 or min_lr_value <= 0.0:
            min_lr_ratios.append(0.0)
        else:
            ratio = min_lr_value / base_lr
            min_lr_ratios.append(float(min(max(ratio, 0.0), 1.0)))

    scheduler_name = str(scheduler_type).strip().lower()

    def make_linear_lambda(min_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return float(min_ratio + (1.0 - min_ratio) * (1.0 - progress))

        return lr_lambda

    def make_cosine_lambda(min_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_ratio + (1.0 - min_ratio) * cosine)

        return lr_lambda

    def make_constant_lambda(_min_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        return lr_lambda

    if scheduler_name == "linear":
        lambdas = [make_linear_lambda(r) for r in min_lr_ratios]
    elif scheduler_name == "cosine":
        lambdas = [make_cosine_lambda(r) for r in min_lr_ratios]
    elif scheduler_name == "constant":
        lambdas = [make_constant_lambda(r) for r in min_lr_ratios]
    else:
        raise ValueError(f"Unsupported lr_scheduler_type={scheduler_type}. Use linear|cosine|constant.")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def maybe_enable_gradient_checkpointing(model: PreTrainedModel) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return None

    if isinstance(model, PeftModel):
        base_model = model.base_model.model
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

    return None


def maybe_compile_model(model: nn.Module, config: "TrainConfig") -> nn.Module:
    if not bool(config.torch_compile):
        return model

    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        print("torch.compile not available; skipping compilation.")
        return model

    compile_kwargs: Dict[str, Any] = {"mode": str(config.torch_compile_mode)}
    signature = inspect.signature(compile_fn)
    if "fullgraph" in signature.parameters:
        compile_kwargs["fullgraph"] = bool(config.torch_compile_fullgraph)
    if "dynamic" in signature.parameters:
        compile_kwargs["dynamic"] = bool(config.torch_compile_dynamic)

    try:
        return compile_fn(model, **compile_kwargs)
    except Exception as exc:
        print(f"torch.compile failed; continuing without compilation. Error: {exc}")
        return model


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str
    output_dir: Path
    batch_size: int
    num_workers: int
    seed: int

    lr: float
    lm_lr: float
    adapter_lr: float
    weight_decay: float
    warmup_ratio: float
    num_epochs: int
    grad_accum_steps: int
    max_grad_norm: float

    lr_scheduler_type: str
    min_lr: float

    optimizer_type: str
    adam_beta1: float
    adam_beta2: float
    adam_eps: float

    max_prompt_tokens: int
    max_target_tokens: int

    lm_lora_r: int
    lm_lora_alpha: int
    lm_lora_dropout: float
    lm_target_modules: str

    num_visual_tokens: int

    projector_layers: int
    projector_hidden_ratio: float
    projector_dropout: float

    adapter_layers: int
    adapter_heads: int
    adapter_attn_dropout: float
    adapter_mlp_ratio: float
    adapter_mlp_dropout: float

    projected_feature_check: bool
    projected_feature_cosine_threshold: float
    projected_feature_max_pairs_to_log: int

    use_mask_template_prompt: bool

    concept_csv_path: Path
    concept_loss_weight: float
    concept_loss_label_smoothing: float
    concept_imbalance_strategy: str
    concept_cb_beta: float
    concept_class_weight_power: float
    concept_max_class_weight: float
    concept_drw_start_epoch: int
    concept_drw_ramp_epochs: int
    concept_balance_across_concepts: bool

    use_concept_balanced_sampler: bool
    concept_sampler_start_epoch: int
    concept_sampler_reduction: str
    concept_sampler_weight_power: float
    concept_sampler_max_weight: float


    contrastive_loss_weight: float
    contrastive_margin_weight: float
    contrastive_temperature: float
    contrastive_margin: float

    index_range: int
    max_videos_per_study: int

    log_every_steps: int
    eval_every_steps: int
    save_every_steps: int

    gen_max_new_tokens: int

    wandb_project: str
    wandb_entity: str
    wandb_run_name: str
    wandb_tags: str
    wandb_mode: str

    attn_implementation: str
    gradient_checkpointing: bool
    lm_head_chunk_size: int
    enable_tf32: bool

    torch_compile: bool
    torch_compile_mode: str
    torch_compile_dynamic: bool
    torch_compile_fullgraph: bool


@dataclass(frozen=True)
class TrainSummary:
    best_step: int
    best_val_loss: float
    best_val_report_acc: float
    final_step: int


