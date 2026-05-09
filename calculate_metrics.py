import json
import re
import numpy as np
from typing import List, Dict, Any, Sequence, Mapping, Tuple, Optional
from pathlib import Path
import argparse
from tqdm import tqdm

# Attempt to import metrics libraries
from sacrebleu.metrics import BLEU
import evaluate
import sacrebleu

def normalize_vlm_prediction(text: str) -> str:
    """
    Converts various VLM output formats to a clean "Section: Content" format matching Ground Truth.
    """
    # --- Format 1: Section name / Content ---
    pattern1 = r"Section name:\s*(.*?)\n[Cc]ontent:\s*"
    text = re.sub(pattern1, r"\1: ", text)
    
    # --- Format 2: Markdown with numbered/bold sections ---
    # Strip preamble (### ...)
    text = re.sub(r"(?m)^###.*?\n", "", text)
    # Strip patient information section completely
    text = re.sub(r"(?i)\*\*Patient Information:\*\*.*?(?=\*\*(\d+\.|\s*[A-Z])|$)", "", text, flags=re.DOTALL)
    
    # Handle bold headers (numbered or not): **1. Left Ventricle:** or **Left Ventricle:**
    # We look for a line starting with bold header
    text = re.sub(r"(?m)^\s*\*\*(?:\d+[\.]?\s*)?(.*?):\*\*\s*", r"\1: ", text)
    
    # Handle sub-bullets headers: "- **Size:**" -> " " (strip header)
    text = re.sub(r"-\s*\*\*(.*?):\*\*\s*", " ", text)
    
    # Remove any remaining bolding stars
    text = text.replace("**", "")
    
    # Remove separators
    text = text.replace("---", "")
    
    # Remove patient info placeholders or trailing incomplete sections
    text = re.sub(r"\[.*?\]", "", text)
    
    # Clean up whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    
    return text.strip()

# --- Metrics Implementations (from test_result_example.py) ---

def tokenize_report_text(text: str) -> List[str]:
    cleaned = str(text).strip()
    if not cleaned: return []
    # Pattern exactly as in example
    pattern = r"<[Mm][Aa][Ss][Kk]>|[A-Za-z0-9]+|[^\sA-Za-z0-9]"
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

class BleuScorer:
    def __init__(self):
        import inspect
        from sacrebleu.metrics import BLEU
        bleu_init_params = inspect.signature(BLEU.__init__).parameters
        bleu_kwargs: Dict[str, Any] = {"smooth_method": "exp"}
        if "effective_order" in bleu_init_params:
            bleu_kwargs["effective_order"] = True
        elif "use_effective_order" in bleu_init_params:
            bleu_kwargs["use_effective_order"] = True

        self._metrics = [
            BLEU(max_ngram_order=i, **bleu_kwargs) for i in range(1, 5)
        ]

    def score_pair(self, reference: str, prediction: str) -> List[float]:
        scores = []
        for metric in self._metrics:
            res = metric.sentence_score(str(prediction), [str(reference)]).score
            scores.append(float(res) / 100.0)
        return scores

class RougeLScorer:
    def __init__(self):
        from rouge_score import rouge_scorer
        # Match use_stemmer=False from example
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def score_pair(self, reference: str, prediction: str) -> float:
        return float(self._scorer.score(str(reference), str(prediction))["rougeL"].fmeasure)

# --- Main Metric Calculation ---

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from Echo results JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to results_*.json")
    parser.add_argument("--test_csv", type=str, help="Optional: path to test.csv to filter IDs")
    parser.add_argument("--no_clean", action="store_true", help="Disable the Section name/Content cleaning regex")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File {input_path} not found.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"--- Dataset Info ---")
    print(f"Total samples found in JSON: {len(data)}")

    # Load test IDs if provided
    test_ids = None
    if args.test_csv:
        csv_path = Path(args.test_csv)
        if csv_path.exists():
            with open(csv_path, "r") as f:
                test_ids = {line.strip() for line in f if line.strip()}
            print(f"Filtering results to {len(test_ids)} test IDs from CSV.")
        else:
            print(f"Warning: test_csv {csv_path} not found. Using all data.")

    # Filter data
    to_eval = []
    for record in data:
        eid = str(record.get("exam_id", ""))
        if test_ids is None or eid in test_ids:
            to_eval.append(record)

    print(f"Samples currently being evaluated: {len(to_eval)}")
    if not to_eval:
        return

    # Prepare lists
    preds = []
    refs = []
    
    for r in to_eval:
        p = r["prediction"]
        # Apply the regex cleaner unless disabled
        if not args.no_clean:
            p = normalize_vlm_prediction(p)
        
        preds.append(p)
        refs.append(r["ground_truth"])

    # Calculate NLP Metrics
    m_results = {}
    
    # BLEU
    print("Computing BLEU (1-4)...")
    bleu_scorer = BleuScorer()
    b_scores = []
    for p, r in zip(preds, refs):
        b_scores.append(bleu_scorer.score_pair(r, p))
    b_means = np.mean(b_scores, axis=0)
    for i in range(4):
        m_results[f"bleu{i+1}"] = b_means[i]

    # ROUGE
    print("Computing ROUGE-L...")
    rouge_scorer = RougeLScorer()
    r_scores = [rouge_scorer.score_pair(r, p) for p, r in zip(preds, refs)]
    m_results["rougeL"] = np.mean(r_scores)

    # METEOR
    print("Computing METEOR...")
    meteor_eval = evaluate.load("meteor")
    meteor_res = meteor_eval.compute(predictions=preds, references=refs)
    m_results["meteor"] = meteor_res["meteor"]

    # CE Metrics (Clinical Entities)
    print("Computing Clinical Entity (CE) Metrics...")
    ce_ps, ce_rs, ce_f1s = [], [], []
    for r, p in zip(refs, preds):
        p_val, r_val, f1_val = compute_ce_precision_recall_f1(r, p)
        ce_ps.append(p_val)
        ce_rs.append(r_val)
        ce_f1s.append(f1_val)
    
    m_results["ce_precision"] = np.mean(ce_ps)
    m_results["ce_recall"] = np.mean(ce_rs)
    m_results["ce_f1"] = np.mean(ce_f1s)

    # Output
    print("\n--- FINAL METRICS ---")
    for k, v in m_results.items():
        print(f"{k}: {v:.4f}")
    print("----------------------")

if __name__ == "__main__":
    main()
