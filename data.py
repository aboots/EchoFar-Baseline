import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from rclstream.datasets.private import echo
import video_utils
import utils
import os
import math
import glob
import json
import pickle

# Third-party library imports s
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pydicom
import sklearn
import sklearn.metrics


# Local module imports
import utils
import video_utils

GT_JSON_PATH = Path("/home/anne/report_gen/findings_token_all.json")

def split_dataset_7_1_2(dataset: Dataset, seed: int) -> Tuple[Subset, Subset, Subset]:
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_set, val_set, test_set


def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, DataLoader]:
    train_set, val_set, test_set = split_dataset_7_1_2(dataset=dataset, seed=seed)

    pin_memory = torch.cuda.is_available()
    return {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _coerce_findings(findings: Any) -> Dict[str, str]:
    if not isinstance(findings, dict):
        return {}

    coerced: Dict[str, str] = {}
    for key, value in findings.items():
        if key is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        value_str = "" if value is None else str(value)
        coerced[key_str] = value_str
    return coerced


def load_findings_by_exam_id(json_path: Path) -> Dict[str, Dict[str, str]]:
    """Load findings JSON and index by exam_id."""
    with json_path.open("r", encoding="utf-8") as f_in:
        raw = json.load(f_in)

    records: List[Dict[str, Any]]
    if isinstance(raw, list):
        records = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        records = [r for r in raw["data"] if isinstance(r, dict)]
    elif isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("exam_id", key)
                records.append(record)
    else:
        records = []

    by_exam_id: Dict[str, Dict[str, str]] = {}
    for record in records:
        exam_id = str(record.get("exam_id", "")).strip()
        if not exam_id:
            continue
        findings = _coerce_findings(record.get("findings", {}))
        if not findings:
            continue
        by_exam_id[exam_id] = findings

    return by_exam_id


def findings_to_report_text(findings: Mapping[str, str]) -> str:
    sections: List[str] = []
    for section_name, section_text in findings.items():
        cleaned_section_name = str(section_name).strip()
        cleaned_section_text = str(section_text).strip()
        if cleaned_section_name and cleaned_section_text:
            sections.append(f"{cleaned_section_name}: {cleaned_section_text}")
    return "\n".join(sections)

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GT_JSON_PATH = Path("/home/mahdi.abootorabi/EchoFAR/findings_token_all.json")
    gt_findings_by_exam_id = load_findings_by_exam_id(GT_JSON_PATH)

    patient_dataset = echo.EchoPatientDataset()
    max_index = min(15000, len(patient_dataset))
    MAX_VIDEOS_PER_STUDY = 5

    for index in range(max_index):
        sample = patient_dataset[index]
        echo_id = sample["exam_id"]
        
        if str(echo_id) not in gt_findings_by_exam_id:
            continue

        videos = sample["videos"]  # List of uint8 numpy arrays with shape (T, H, W)
        
        if not videos:
            continue

        # Take up to 5 videos
        video_list = videos[:MAX_VIDEOS_PER_STUDY]
        print(f"Exam ID {echo_id}: Found {len(video_list)} videos. Ready for inference.")
        
        # Here you would call your model inference
        # For example: report = generate_report(video_list)

if __name__ == "__main__":
    main()

