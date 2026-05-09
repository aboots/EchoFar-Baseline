from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from qwen_vl_utils import process_vision_info
from rclstream.datasets.private import echo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GT_JSON_PATH = Path("/home/mahdi.abootorabi/EchoFAR/findings_token_all.json")
TRAIN_CSV_PATH = Path("/home/mahdi.abootorabi/EchoFAR/data/train.csv")
VAL_CSV_PATH = Path("/home/mahdi.abootorabi/EchoFAR/data/val.csv")

MAX_VIDEOS_PER_STUDY = 5
MAX_FRAMES_PER_VIDEO = 16

SYSTEM_PROMPT = (
    "You are a medical assistant generating an echocardiography findings report.\n"
    "Generate a detailed report based on the provided echocardiography videos. "
    "Write one section per line in the format 'Section name: content'.\n"
    "Required sections: Left Ventricle, Right Ventricle, Left Atrium, Right Atrium, "
    "Mitral Valve, Tricuspid Valve, Aortic Valve, Pulmonary Valve/Artery, Aorta, "
    "Venous, Pericardium/Other."
)

PROMPT_SUFFIX = "\n\nBased on the echocardiography videos above, generate the findings report:"


# ---------------------------------------------------------------------------
# Video utilities  (same as inference scripts)
# ---------------------------------------------------------------------------

def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x_float = x.astype(np.float32)
    x_min, x_max = float(np.nanmin(x_float)), float(np.nanmax(x_float))
    x_scaled = x_float * 255.0 if (x_max <= 1.0 and x_min >= 0.0) else x_float
    return np.clip(x_scaled, 0.0, 255.0).astype(np.uint8)


def video_thw_to_thwc_rgb_uint8(video_thw: np.ndarray) -> np.ndarray:
    if video_thw.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got {video_thw.shape}.")
    return np.repeat(_to_uint8(video_thw)[..., None], 3, axis=-1)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_exam_ids_from_csv(csv_path: Path) -> List[str]:
    with open(csv_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def findings_to_report_text(findings: Mapping[str, str]) -> str:
    sections = []
    for k, v in findings.items():
        k_clean, v_clean = str(k).strip(), str(v).strip()
        if k_clean and v_clean:
            sections.append(f"{k_clean}: {v_clean}")
    return "\n".join(sections)


def load_findings_by_exam_id(json_path: Path) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result: Dict[str, str] = {}
    for record in raw:
        exam_id = str(record.get("exam_id", "")).strip()
        findings = record.get("findings", {})
        if exam_id and findings:
            result[exam_id] = findings_to_report_text(findings)
    return result


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EchoFinetuneDataset(Dataset):
    def __init__(
        self,
        exam_ids: List[str],
        patient_dataset: Any,
        gt_reports: Dict[str, str],
        exam_id_to_idx: Dict[str, int],
    ) -> None:
        self.patient_dataset = patient_dataset
        self.gt_reports = gt_reports
        self.exam_id_to_idx = exam_id_to_idx
        self.exam_ids = [
            eid for eid in exam_ids
            if eid in exam_id_to_idx and eid in gt_reports
        ]
        print(f"  {len(self.exam_ids)} valid samples (from {len(exam_ids)} IDs in CSV)")

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        exam_id = self.exam_ids[idx]
        ds_idx = self.exam_id_to_idx[exam_id]
        sample = self.patient_dataset[ds_idx]

        videos: List[np.ndarray] = []
        for v in sample["videos"][:MAX_VIDEOS_PER_STUDY]:
            v_uint8 = video_thw_to_thwc_rgb_uint8(v)
            t = v_uint8.shape[0]
            if t > MAX_FRAMES_PER_VIDEO:
                indices = np.linspace(0, t - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
                v_uint8 = v_uint8[indices]
            videos.append(v_uint8)

        return {
            "exam_id": exam_id,
            "videos": videos,
            "report": self.gt_reports[exam_id],
        }


# ---------------------------------------------------------------------------
# Collator  — runs Qwen2.5-VL processor and creates masked labels
# ---------------------------------------------------------------------------

class QoQFinetuneCollator:
    """
    Builds the full user+assistant message, processes it through the Qwen2.5-VL
    processor, and masks all prompt tokens in the labels so the loss is computed
    only on the report tokens.
    """

    def __init__(self, processor: Any, system_prompt: str, prompt_suffix: str) -> None:
        self.processor = processor
        self.system_prompt = system_prompt
        self.prompt_suffix = prompt_suffix

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert len(batch) == 1, (
            "batch_size must be 1; use grad_accum_steps to control effective batch size"
        )
        item = batch[0]

        # Build user content: system prompt + videos + prompt suffix
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self.system_prompt + "\n\nEchocardiography Videos:\n"},
        ]
        for j, v_arr in enumerate(item["videos"]):
            content.append({"type": "text", "text": f"[Video {j + 1}]: "})
            content.append({
                "type": "video",
                "video": [Image.fromarray(frame) for frame in v_arr],
                "fps": 1.0,
            })
            content.append({"type": "text", "text": " "})
        content.append({"type": "text", "text": self.prompt_suffix})

        messages_full = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": item["report"]},
        ]

        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages_full)

        inputs = self.processor(
            text=[text_full],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
        )

        # Mask all prompt tokens; only compute loss on the report (assistant) tokens
        labels = inputs.input_ids.clone()
        response_start = self._find_response_start(inputs.input_ids[0])
        labels[0, :response_start] = -100
        inputs["labels"] = labels

        return dict(inputs)

    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        """
        Locates the first token of the assistant response in the tokenized sequence
        by finding the last '<|im_start|>assistant\\n' pattern.
        """
        tokenizer = self.processor.tokenizer
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False)
        header_len = len(assistant_header)

        ids = input_ids.tolist()
        for i in range(len(ids) - header_len - 1, -1, -1):
            if ids[i] == im_start_id and ids[i + 1: i + 1 + header_len] == assistant_header:
                return i + 1 + header_len

        print("Warning: could not find assistant response boundary; labels fully masked.")
        return len(ids)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def freeze_vision_backbone(model: nn.Module) -> None:
    frozen = 0
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad_(False)
            frozen += param.numel()
    print(f"Frozen {frozen:,} vision backbone parameters.")


def apply_lora(
    model: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> nn.Module:
    """
    Applies LoRA to the LLM attention layers only.
    Qwen2.5-VL's vision encoder uses fused 'qkv', so the target modules
    ['q_proj','k_proj','v_proj','o_proj'] hit only the LLM transformer layers.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
) -> torch.optim.AdamW:
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups, lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir: Path, step: int, model: nn.Module, processor: Any) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))
    print(f"Saved checkpoint → {ckpt_dir}")


def save_best_checkpoint(
    output_dir: Path,
    model: nn.Module,
    processor: Any,
    val_loss: float,
    step: int,
) -> None:
    best_dir = output_dir / "checkpoint-best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    processor.save_pretrained(str(best_dir))
    with open(best_dir / "meta.json", "w") as f:
        json.dump({"val_loss": val_loss, "step": step}, f, indent=2)
    print(f"Saved best checkpoint (val_loss={val_loss:.4f}, step={step}) → {best_dir}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_steps: int = 200,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        if i >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True):
            out = model(**batch)
        loss = out.loss
        if torch.isfinite(loss):
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16

    # --- Load model and processor ---
    print(f"Loading model: {args.model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=autocast_dtype,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    # --- Freeze vision, apply LoRA to LLM layers ---
    freeze_vision_backbone(model)
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model = apply_lora(
        model=model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
    )

    # --- Load data ---
    print("Loading dataset and ground-truth reports...")
    patient_dataset = echo.EchoPatientDataset()
    gt_reports = load_findings_by_exam_id(GT_JSON_PATH)
    exam_id_to_idx = {
        str(row["exam_id"]): i
        for i, row in patient_dataset.patient_metadata.iterrows()
    }

    print("Train split:")
    train_dataset = EchoFinetuneDataset(
        load_exam_ids_from_csv(TRAIN_CSV_PATH), patient_dataset, gt_reports, exam_id_to_idx
    )
    print("Val split:")
    val_dataset = EchoFinetuneDataset(
        load_exam_ids_from_csv(VAL_CSV_PATH), patient_dataset, gt_reports, exam_id_to_idx
    )

    collator = QoQFinetuneCollator(processor, SYSTEM_PROMPT, PROMPT_SUFFIX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    # --- Optimizer & scheduler ---
    optimizer = create_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)

    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(
        f"\nTraining config: epochs={args.num_epochs} steps/epoch={steps_per_epoch} "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"effective_batch={args.grad_accum_steps}\n"
    )

    # --- Training loop ---
    global_step = 0
    best_val_loss = float("inf")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        interval_loss_sum = 0.0
        interval_loss_count = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        ):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True):
                out = model(**batch)
                loss = out.loss / args.grad_accum_steps

            loss.backward()

            raw_loss = out.loss.detach().item()
            if torch.isfinite(out.loss):
                interval_loss_sum += raw_loss
                interval_loss_count += 1
                epoch_loss_sum += raw_loss
                epoch_loss_count += 1

            is_accum_step = (step + 1) % args.grad_accum_steps == 0
            is_last_batch = (step + 1) == len(train_loader)

            if not (is_accum_step or is_last_batch):
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.max_grad_norm,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every_steps == 0:
                avg_loss = interval_loss_sum / max(1, interval_loss_count)
                lr_val = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"train_loss={avg_loss:.4f} lr={lr_val:.2e}"
                )
                interval_loss_sum = 0.0
                interval_loss_count = 0

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(args.output_dir, global_step, model, processor)

        # --- End-of-epoch validation ---
        avg_epoch_train_loss = epoch_loss_sum / max(1, epoch_loss_count)
        print(f"\nEpoch {epoch + 1} finished — avg_train_loss={avg_epoch_train_loss:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("Running validation...")
        val_loss = run_val_loss(
            model, val_loader, device, autocast_dtype, max_steps=args.val_max_steps
        )
        print(f"Epoch {epoch + 1} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(args.output_dir, model, processor, val_loss, global_step)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Final checkpoint ---
    save_checkpoint(args.output_dir, global_step, model, processor)
    print(f"\nTraining complete.")
    print(f"Best val_loss={best_val_loss:.4f}")
    print(f"Best model: {args.output_dir / 'checkpoint-best'}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune QoQ-Med-VL-7B with LoRA (frozen vision backbone)"
    )

    # Paths
    parser.add_argument(
        "--model_name_or_path", type=str, default="ddvd233/QoQ-Med-VL-7B"
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./qoq_med_finetuned"))

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Effective batch size = grad_accum_steps (batch_size is always 1)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Optimizer
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear"]
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of LLM attention modules to apply LoRA to",
    )

    # Logging & saving
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument(
        "--val_max_steps", type=int, default=200,
        help="Max validation batches per epoch (cap to save time)"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
