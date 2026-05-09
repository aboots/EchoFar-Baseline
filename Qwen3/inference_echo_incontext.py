import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from rclstream.datasets.private import echo
import numpy as np
from typing import List, Dict, Any, Sequence, Mapping
import json
from pathlib import Path
from tqdm import tqdm
import evaluate
import os

# Paths
GT_JSON_PATH = Path("/home/mahdi.abootorabi/EchoFAR/findings_token_all.json")
OUTPUT_PATH = Path("/home/mahdi.abootorabi/EchoFAR/results_qwen3_echo_incontext.json")
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
TEST_CSV_PATH = Path("/home/mahdi.abootorabi/EchoFAR/data/test.csv")
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

# --- Medical Metric Utils (from test_result_example.py) ---
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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Load dataset and ground truth
    print("Loading datasets...")
    patient_dataset = echo.EchoPatientDataset()
    gt_reports_by_exam_id = load_findings_by_exam_id(GT_JSON_PATH)
    
    system_prompt = (
        "You are a medical assistant generating an echocardiography findings report.\n"
        "Write one section per line in the format 'Section name: content'.\n"
    )
    prompt_body = "Generate an echocardiography findings report for the provided target videos, following the format and style of the examples above."
    
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

    # --- Pre-load Visual ICL Examples ---
    print("Pre-loading Visual ICL examples (ID 138 and ID 9)...")
    ic_examples = []
    for eid in ["138", "9"]:
        if eid not in exam_id_to_idx:
            print(f"Warning: ICL ID {eid} not found in metadata.")
            continue
        idx = exam_id_to_idx[eid]
        sample = patient_dataset[idx]
        v_list = sample["videos"][:MAX_VIDEOS_PER_STUDY]
        
        processed_v = []
        for v in v_list:
            v_uint8 = video_thw_to_thwc_rgb_uint8(v)
            t = v_uint8.shape[0]
            if t > MAX_FRAMES_PER_VIDEO:
                indices = np.linspace(0, t - 1, MAX_FRAMES_PER_VIDEO, dtype=int)
                v_uint8 = v_uint8[indices]
            processed_v.append(v_uint8)
        
        ic_examples.append({
            "id": eid,
            "videos": processed_v,
            "report": gt_reports_by_exam_id.get(eid, "")
        })

    # Filter test_ids to those present in the dataset and not yet processed
    to_process_ids = [eid for eid in test_ids if eid in exam_id_to_idx and eid not in processed_exam_ids]
    print(f"Total samples to process: {len(to_process_ids)} (skipping {len(test_ids) - len(to_process_ids)} already processed or missing)")

    for exam_id in tqdm(to_process_ids, desc="Generating reports"):
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
            
            # --- Construct Highly Structured ICL Message ---
            content = [{"type": "text", "text": system_prompt + "\n\n"}]
            all_videos = []

            for i, ex in enumerate(ic_examples):
                content.append({"type": "text", "text": f"--- DEMONSTRATION EXAMPLE {i+1} ---\n"})
                content.append({"type": "text", "text": f"Input Videos for Demonstration {i+1}:\n"})
                for j in range(len(ex["videos"])):
                    content.append({"type": "text", "text": f"[Example {i+1}, Video {j+1}]: "})
                    content.append({"type": "video"})
                    all_videos.append(ex["videos"][j])
                    content.append({"type": "text", "text": " "})
                
                content.append({"type": "text", "text": f"\n\nOutput Findings Report for Demonstration {i+1}:\n{ex['report']}\n"})
                content.append({"type": "text", "text": f"--- END OF DEMONSTRATION {i+1} ---\n\n"})

            content.append({"type": "text", "text": "#########################################\n"})
            content.append({"type": "text", "text": "--- TARGET INFERENCE TASK ---\n"})
            content.append({"type": "text", "text": "Task: Analyze the following 5 target echocardiography videos and generate a findings report.\n"})
            content.append({"type": "text", "text": "Target Videos for Analysis:\n"})
            for j in range(len(sampled_target_videos)):
                content.append({"type": "text", "text": f"[Target Video {j+1}]: "})
                content.append({"type": "video"})
                all_videos.append(sampled_target_videos[j])
                content.append({"type": "text", "text": " "})

            content.append({"type": "text", "text": "\n\nFinal Report for Target Case (matching the clinical style of the demonstrations):\nReport:"})

            messages = [{"role": "user", "content": content}]

            # video_metadata = {
            #     "fps": 1.0,
            #     "total_num_frames": MAX_FRAMES_PER_VIDEO
            # }

            # Define metadata for all videos (training examples + target)
            # This satisfies the processor's requirement for temporal information
            video_metadata = [
                {"fps": 1.0, "total_num_frames": v.shape[0]} 
                for v in all_videos
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            inputs = processor(
                text=[text], 
                videos=all_videos, 
                video_metadata=video_metadata,
                padding=True, 
                return_tensors="pt"
            ).to(model.device)

            output_ids = model.generate(**inputs, max_new_tokens=512)
            trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            results.append({
                "exam_id": exam_id,
                "prediction": decoded,
                "ground_truth": gt_reports_by_exam_id[exam_id]
            })
            
            # Save every 10 samples to avoid data loss
            if len(results) % 10 == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Compute Metrics ONLY for test set IDs
    print("\nComputing metrics...")
    # Filter results to only include those in test_ids
    test_results = [r for r in results if str(r["exam_id"]) in test_ids]
    
    preds = [r["prediction"] for r in test_results]
    refs = [r["ground_truth"] for r in test_results]

    if not preds:
        print("No predictions to evaluate.")
        return

    m_results = {}
    
    # BLEU 1-4
    bleu_eval = evaluate.load("bleu")
    for i in range(1, 5):
        try:
            res = bleu_eval.compute(predictions=preds, references=[[r] for r in refs], max_order=i)
            m_results[f"test_bleu{i}"] = res["bleu"]
        except:
            m_results[f"test_bleu{i}"] = 0.0

    # ROUGE-L
    rouge_eval = evaluate.load("rouge")
    res = rouge_eval.compute(predictions=preds, references=refs)
    m_results["test_rougeL"] = res["rougeL"]

    # METEOR
    try:
        meteor_eval = evaluate.load("meteor")
        res = meteor_eval.compute(predictions=preds, references=refs)
        m_results["test_meteor"] = res["meteor"]
    except:
        m_results["test_meteor"] = 0.0

    # CIDEr
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

    # Print final metrics
    metrics_str = " ".join([f"{k}={v:.4f}" for k, v in m_results.items()])
    print("\n--- Final Metrics ---")
    print(metrics_str)
    print("----------------------")

if __name__ == "__main__":
    main()
