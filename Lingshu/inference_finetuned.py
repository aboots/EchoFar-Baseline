import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from rclstream.datasets.private import echo
import numpy as np
from typing import List, Dict, Any, Sequence, Mapping, Tuple
import json
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import evaluate
import argparse

# Paths
GT_JSON_PATH = Path("/home/mahdi.abootorabi/EchoFAR/findings_token_all.json")
TEST_CSV_PATH = Path("/home/mahdi.abootorabi/EchoFAR/data/test.csv")
BASE_MODEL_ID = "lingshu-medical-mllm/Lingshu-7B"
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


# --- Video utils ---
def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x_float = x.astype(np.float32)
    x_min, x_max = float(np.nanmin(x_float)), float(np.nanmax(x_float))
    x_scaled = x_float * 255.0 if (x_max <= 1.0 and x_min >= 0.0) else x_float
    return np.clip(x_scaled, 0.0, 255.0).astype(np.uint8)

def video_thw_to_thwc_rgb_uint8(video_thw: np.ndarray) -> np.ndarray:
    if video_thw.ndim != 3:
        raise ValueError(f"Expected (T, H, W). Got shape={video_thw.shape}.")
    return np.repeat(_to_uint8(video_thw)[..., None], 3, axis=-1)


# --- Data utils ---
def findings_to_report_text(findings: Mapping[str, str]) -> str:
    sections = []
    for k, v in findings.items():
        k_clean, v_clean = str(k).strip(), str(v).strip()
        if k_clean and v_clean:
            sections.append(f"{k_clean}: {v_clean}")
    return "\n".join(sections)

def load_findings_by_exam_id(json_path: Path) -> Dict[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    result: Dict[str, str] = {}
    for record in raw:
        exam_id = str(record.get("exam_id", "")).strip()
        findings = record.get("findings", {})
        if exam_id and findings:
            result[exam_id] = findings_to_report_text(findings)
    return result


# --- Metrics ---
def tokenize_report_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", str(text).strip())

def normalize_tokens_for_scoring(tokens: Sequence[str]) -> List[str]:
    return [str(t).strip().lower() for t in tokens if str(t).strip()]

def compute_ce_precision_recall_f1(gt_report: str, generated_report: str) -> Tuple[float, float, float]:
    gt_tokens = set(normalize_tokens_for_scoring(tokenize_report_text(gt_report)))
    pred_tokens = set(normalize_tokens_for_scoring(tokenize_report_text(generated_report)))
    if not gt_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0
    tp = len(gt_tokens & pred_tokens)
    p = float(tp) / float(len(pred_tokens))
    r = float(tp) / float(len(gt_tokens))
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser(description="Inference with finetuned Lingshu on test set")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/home/mahdi.abootorabi/EchoFAR/lingshu_finetuned/checkpoint-best",
        help="Path to the saved LoRA adapter (checkpoint-best or any checkpoint-N dir)",
    )
    parser.add_argument(
        "--base_model", type=str, default=BASE_MODEL_ID,
        help="HuggingFace model ID or local path for the base model",
    )
    parser.add_argument(
        "--output_path", type=str,
        default="/home/mahdi.abootorabi/EchoFAR/results_lingshu_finetuned.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)

    # --- Load base model + LoRA adapter ---
    print(f"Loading base model: {args.base_model}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.adapter_path)

    # --- Load dataset and GT ---
    print("Loading datasets...")
    patient_dataset = echo.EchoPatientDataset()
    gt_reports_by_exam_id = load_findings_by_exam_id(GT_JSON_PATH)

    print(f"Loading test IDs from {TEST_CSV_PATH}...")
    with open(TEST_CSV_PATH, "r") as f:
        test_ids = {line.strip() for line in f if line.strip()}
    print(f"Found {len(test_ids)} test IDs.")

    exam_id_to_idx = {
        str(row["exam_id"]): i
        for i, row in patient_dataset.patient_metadata.iterrows()
    }

    # Resume from existing output if present
    results = []
    processed_exam_ids: set = set()
    if output_path.exists():
        print(f"Loading existing results from {output_path}...")
        with open(output_path, "r") as f:
            results = json.load(f)
        processed_exam_ids = {str(r["exam_id"]) for r in results}
        print(f"Loaded {len(processed_exam_ids)} existing results.")

    to_process_ids = [
        eid for eid in test_ids
        if eid in exam_id_to_idx and eid not in processed_exam_ids
    ]
    print(f"Samples to process: {len(to_process_ids)}")

    for exam_id in tqdm(to_process_ids, desc="Finetuned Lingshu inference"):
        try:
            idx = exam_id_to_idx[exam_id]
            sample = patient_dataset[idx]

            if exam_id not in gt_reports_by_exam_id:
                continue

            target_videos = sample["videos"][:MAX_VIDEOS_PER_STUDY]
            if not target_videos:
                continue

            sampled_videos = []
            for v in target_videos:
                v_uint8 = video_thw_to_thwc_rgb_uint8(v)
                t = v_uint8.shape[0]
                if t > MAX_FRAMES_PER_VIDEO:
                    indices = np.linspace(0, t - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
                    v_uint8 = v_uint8[indices]
                sampled_videos.append(v_uint8)

            # Build prompt (same format as finetuning collator)
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": SYSTEM_PROMPT + "\n\nEchocardiography Videos:\n"},
            ]
            for j, v_arr in enumerate(sampled_videos):
                content.append({"type": "text", "text": f"[Video {j + 1}]: "})
                content.append({
                    "type": "video",
                    "video": [Image.fromarray(frame) for frame in v_arr],
                    "fps": 1.0,
                })
                content.append({"type": "text", "text": " "})
            content.append({"type": "text", "text": PROMPT_SUFFIX})

            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512)
            trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            decoded = processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            results.append({
                "exam_id": exam_id,
                "prediction": decoded,
                "ground_truth": gt_reports_by_exam_id[exam_id],
            })

            if len(results) % 10 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error processing {exam_id}: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")

    # --- Compute metrics ---
    print("\nComputing metrics...")
    test_results = [r for r in results if str(r["exam_id"]) in test_ids]
    if not test_results:
        print("No predictions to evaluate.")
        return

    preds = [r["prediction"] for r in test_results]
    refs = [r["ground_truth"] for r in test_results]

    m_results = {}

    bleu_eval = evaluate.load("bleu")
    for i in range(1, 5):
        try:
            res = bleu_eval.compute(predictions=preds, references=[[r] for r in refs], max_order=i)
            m_results[f"test_bleu{i}"] = res["bleu"]
        except:
            m_results[f"test_bleu{i}"] = 0.0

    rouge_eval = evaluate.load("rouge")
    res = rouge_eval.compute(predictions=preds, references=refs)
    m_results["test_rougeL"] = res["rougeL"]

    try:
        meteor_eval = evaluate.load("meteor")
        res = meteor_eval.compute(predictions=preds, references=refs)
        m_results["test_meteor"] = res["meteor"]
    except:
        m_results["test_meteor"] = 0.0

    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        hypo = {i: [p] for i, p in enumerate(preds)}
        ref = {i: [r] for i, r in enumerate(refs)}
        score, _ = scorer.compute_score(ref, hypo)
        m_results["test_cider"] = score
    except Exception as e:
        print(f"CIDEr error: {e}")
        m_results["test_cider"] = 0.0

    ce_ps, ce_rs, ce_f1s = [], [], []
    for r, p in zip(refs, preds):
        p_val, r_val, f1_val = compute_ce_precision_recall_f1(r, p)
        ce_ps.append(p_val)
        ce_rs.append(r_val)
        ce_f1s.append(f1_val)

    m_results["test_ce_p"] = float(np.mean(ce_ps))
    m_results["test_ce_r"] = float(np.mean(ce_rs))
    m_results["test_ce_f1"] = float(np.mean(ce_f1s))

    print("\n--- Finetuned Lingshu Test Metrics ---")
    for k, v in m_results.items():
        print(f"  {k}: {v:.4f}")
    print("--------------------------------------")

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(m_results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
