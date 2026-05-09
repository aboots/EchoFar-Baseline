import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from rclstream.datasets.private import echo
import numpy as np
from typing import List, Dict, Any, Sequence, Mapping
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import evaluate
import os

# Paths
GT_JSON_PATH = Path("/home/mahdi.abootorabi/EchoFAR/findings_token_all.json")
OUTPUT_PATH = Path("/home/mahdi.abootorabi/EchoFAR/results_lingshu_zeroshot.json")
TEST_CSV_PATH = Path("/home/mahdi.abootorabi/EchoFAR/data/test.csv")
MODEL_ID = "lingshu-medical-mllm/Lingshu-7B"
MAX_VIDEOS_PER_STUDY = 5
MAX_FRAMES_PER_VIDEO = 16 # Take 16 frames uniformly from each video

def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x_float = x.astype(np.float32)
    x_min = float(np.nanmin(x_float))
    x_max = float(np.nanmax(x_float))
    if x_max <= 1.0 and x_min >= 0.0:
        x_scaled = x_float * 255.0
    else:
        x_scaled = x_float
    x_clipped = np.clip(x_scaled, 0.0, 255.0)
    return x_clipped.astype(np.uint8)

def video_thw_to_thwc_rgb_uint8(video_thw: np.ndarray) -> np.ndarray:
    if video_thw.ndim != 3:
        raise ValueError(f"Expected (T, H, W). Got shape={video_thw.shape}.")
    video_uint8 = _to_uint8(video_thw)
    video_thw1 = video_uint8[..., None]
    video_thwc = np.repeat(video_thw1, repeats=3, axis=-1)
    return video_thwc

def findings_to_report_text(findings: Mapping[str, str]) -> str:
    sections: List[str] = []
    for section_name, section_text in findings.items():
        cleaned_section_name = str(section_name).strip()
        cleaned_section_text = str(section_text).strip()
        if cleaned_section_name and cleaned_section_text:
            sections.append(f"{cleaned_section_name}: {cleaned_section_text}")
    return "\n".join(sections)

def load_findings_by_exam_id(json_path: Path) -> Dict[str, str]:
    with json_path.open("r", encoding="utf-8") as f_in:
        raw = json.load(f_in)
    
    by_exam_id: Dict[str, str] = {}
    for record in raw:
        exam_id = str(record.get("exam_id", "")).strip()
        findings = record.get("findings", {})
        if exam_id and findings:
            by_exam_id[exam_id] = findings_to_report_text(findings)
    return by_exam_id

# --- Medical Metric Utils ---
import re
from typing import Tuple, Sequence

def tokenize_report_text(text: str) -> List[str]:
    cleaned = str(text).strip()
    if not cleaned: return []
    pattern = r"[A-Za-z0-9]+|[^\sA-Za-z0-9]"
    return re.findall(pattern, cleaned)

def normalize_report_token(token: str) -> str:
    return str(token).strip().lower()

def normalize_tokens_for_scoring(tokens: Sequence[str], ignored_tokens: Sequence[str] = ()) -> List[str]:
    ignored = {normalize_report_token(t) for t in ignored_tokens}
    normalized: List[str] = []
    for token in tokens:
        token_norm = normalize_report_token(token)
        if not token_norm: continue
        if token_norm in ignored: continue
        normalized.append(token_norm)
    return normalized

def compute_ce_precision_recall_f1(gt_report: str, generated_report: str) -> Tuple[float, float, float]:
    gt_tokens = set(normalize_tokens_for_scoring(tokenize_report_text(gt_report)))
    pred_tokens = set(normalize_tokens_for_scoring(tokenize_report_text(generated_report)))
    if not gt_tokens and not pred_tokens: return 1.0, 1.0, 1.0
    if not gt_tokens or not pred_tokens: return 0.0, 0.0, 0.0
    tp = len(gt_tokens & pred_tokens)
    p = float(tp) / float(len(pred_tokens)) if pred_tokens else 0.0
    r = float(tp) / float(len(gt_tokens)) if gt_tokens else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
    return p, r, f1

def main():
    # Load model and processor
    print(f"Loading model {MODEL_ID}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Load dataset and ground truth
    print("Loading datasets...")
    patient_dataset = echo.EchoPatientDataset()
    gt_reports_by_exam_id = load_findings_by_exam_id(GT_JSON_PATH)
    
    # Zero-Shot Prompt Construction
    system_prompt = (
        "You are a medical assistant generating an echocardiography findings report.\n"
        "Generate a detailed report based on the provided videos. Write one section per line.\n"
        "Required Format (Section name: content):\n"
        "Left Ventricle: content\n"
        "Right Ventricle: content\n"
        "Left Atrium: content\n"
        "Right Atrium: content\n"
        "Mitral Valve: content\n"
        "Tricuspid Valve: content\n"
        "Aortic Valve: content\n"
        "Pulmonary Valve/Artery: content\n"
        "Aorta: content\n"
        "Venous: content\n"
        "Pericardium/Other: content"
    )

    # Load test IDs
    print(f"Loading test IDs from {TEST_CSV_PATH}...")
    with open(TEST_CSV_PATH, "r") as f:
        test_ids = {line.strip() for line in f if line.strip()}
    print(f"Found {len(test_ids)} test IDs.")

    results = []
    processed_exam_ids = set()
    if OUTPUT_PATH.exists():
        print(f"Loading existing results from {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, "r") as f:
            results = json.load(f)
            processed_exam_ids = {str(r["exam_id"]) for r in results}
        print(f"Loaded {len(processed_exam_ids)} existing results.")

    # Create mapping from exam_id to index in patient_dataset
    exam_id_to_idx = {str(row["exam_id"]): i for i, row in patient_dataset.patient_metadata.iterrows()}

    # Filter test_ids to those present in the dataset and not yet processed
    to_process_ids = [eid for eid in test_ids if eid in exam_id_to_idx and eid not in processed_exam_ids]
    print(f"Total samples to process: {len(to_process_ids)} (skipping {len(test_ids) - len(to_process_ids)} already processed or missing)")

    for exam_id in tqdm(to_process_ids, desc="Zero-Shot Generation"):
        try:
            idx = exam_id_to_idx[exam_id]
            sample = patient_dataset[idx]

            if exam_id not in gt_reports_by_exam_id:
                continue

            target_videos = sample["videos"][:MAX_VIDEOS_PER_STUDY]
            if not target_videos:
                continue

            # Sample frames for target videos
            sampled_target_videos = []
            for v in target_videos:
                v_uint8 = video_thw_to_thwc_rgb_uint8(v)
                t = v_uint8.shape[0]
                if t > MAX_FRAMES_PER_VIDEO:
                    indices = np.linspace(0, t - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
                    v_uint8 = v_uint8[indices]
                sampled_target_videos.append(v_uint8)
            
            # --- Construct Message for Lingshu ---
            content = [{"type": "text", "text": system_prompt + "\n\nTarget Videos for Analysis:\n"}]
            for j, v_arr in enumerate(sampled_target_videos):
                content.append({"type": "text", "text": f"[Target Video {j+1}]: "})
                content.append({"type": "video", "video": [Image.fromarray(f) for f in v_arr], "fps": 1.0})
                content.append({"type": "text", "text": " "})

            content.append({"type": "text", "text": "\n\nFinal Report for Target Case:\nReport:"})
            messages = [{"role": "user", "content": content}]

            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            output_ids = model.generate(**inputs, max_new_tokens=512)
            trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            results.append({
                "exam_id": exam_id,
                "prediction": decoded,
                "ground_truth": gt_reports_by_exam_id[exam_id]
            })
            
            if len(results) % 10 == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error processing {exam_id}: {e}")
            continue

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Compute Metrics
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
        except: m_results[f"test_bleu{i}"] = 0.0

    rouge_eval = evaluate.load("rouge")
    res = rouge_eval.compute(predictions=preds, references=refs)
    m_results["test_rougeL"] = res["rougeL"]

    try:
        meteor_eval = evaluate.load("meteor")
        res = meteor_eval.compute(predictions=preds, references=refs)
        m_results["test_meteor"] = res["meteor"]
    except: m_results["test_meteor"] = 0.0

    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        hypo = {i: [p] for i, p in enumerate(preds)}
        ref = {i: [r] for i, r in enumerate(refs)}
        score, _ = scorer.compute_score(ref, hypo)
        m_results["test_cider"] = score
    except: m_results["test_cider"] = 0.0

    # CE Metrics (Clinical Entity overlap)
    ce_ps, ce_rs, ce_f1s = [], [], []
    for r, p in zip(refs, preds):
        p_val, r_val, f1_val = compute_ce_precision_recall_f1(r, p)
        ce_ps.append(p_val)
        ce_rs.append(r_val)
        ce_f1s.append(f1_val)
    
    m_results["test_ce_p"] = np.mean(ce_ps)
    m_results["test_ce_r"] = np.mean(ce_rs)
    m_results["test_ce_f1"] = np.mean(ce_f1s)
    m_results["test_bleurt"] = 0.0

    metrics_str = " ".join([f"{k}={v:.4f}" for k, v in m_results.items()])
    print("\n--- Final Zero-Shot Metrics ---")
    print(metrics_str)
    print("--------------------------------")

if __name__ == "__main__":
    main()
