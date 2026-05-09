from __future__ import annotations

from .common_doppler import *

def tokenize_report_text(text: str) -> List[str]:
    cleaned = str(text).strip()
    if not cleaned:
        return []
    pattern = r"<[Mm][Aa][Ss][Kk]>|[A-Za-z0-9]+|[^\sA-Za-z0-9]"
    return re.findall(pattern, cleaned)


def normalize_report_token(token: str) -> str:
    return str(token).strip().lower()


def normalize_tokens_for_scoring(tokens: Sequence[str], ignored_tokens: Sequence[str]) -> List[str]:
    ignored = {normalize_report_token(t) for t in ignored_tokens}
    normalized: List[str] = []
    for token in tokens:
        token_norm = normalize_report_token(token)
        if not token_norm:
            continue
        if token_norm in ignored:
            continue
        normalized.append(token_norm)
    return normalized


def compute_report_word_accuracy(
    gt_report: str,
    predicted_report: str,
    ignored_tokens: Sequence[str] = ("<MASK>",),
) -> Tuple[int, int]:
    """Compute token accuracy via alignment, ignoring <MASK> (recall-like)."""
    gt_tokens = normalize_tokens_for_scoring(tokens=tokenize_report_text(gt_report), ignored_tokens=ignored_tokens)
    pred_tokens = normalize_tokens_for_scoring(
        tokens=tokenize_report_text(predicted_report), ignored_tokens=ignored_tokens
    )

    total = int(len(gt_tokens))
    if total == 0:
        return 0, 0

    matcher = difflib.SequenceMatcher(a=gt_tokens, b=pred_tokens, autojunk=False)
    correct = int(sum(block.size for block in matcher.get_matching_blocks()))
    return correct, total


def _extract_after_last_marker(text: str, marker: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    idx = cleaned.rfind(str(marker))
    if idx < 0:
        return cleaned
    return cleaned[idx + len(str(marker)) :].strip()


def _extract_report_body_from_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    report_start_tag = "<report>"
    report_end_tag = "</report>"
    start_idx = cleaned.find(report_start_tag)
    if start_idx >= 0:
        end_idx = cleaned.find(report_end_tag, start_idx + len(report_start_tag))
        if end_idx > start_idx:
            return cleaned[start_idx + len(report_start_tag) : end_idx].strip()

    return _extract_after_last_marker(cleaned, "Report:")


MODEL_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+\|>")
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
REASONING_BLOCK_PATTERN = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    flags=re.DOTALL | re.IGNORECASE,
)


def clean_decoded_generation_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\u200b", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    cleaned = MODEL_SPECIAL_TOKEN_PATTERN.sub("", cleaned)

    cleaned = cleaned.replace("<s>", "").replace("</s>", "")
    cleaned = cleaned.replace("<pad>", "").replace("</pad>", "")

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def _extract_reasoning_blocks(text: str) -> str:
    segments: List[str] = []

    for match in THINK_BLOCK_PATTERN.finditer(text):
        segment = str(match.group(1)).strip()
        if segment:
            segments.append(segment)

    for match in REASONING_BLOCK_PATTERN.finditer(text):
        segment = str(match.group(1)).strip()
        if segment:
            segments.append(segment)

    return "\n\n".join(segments).strip()


def _remove_reasoning_blocks(text: str) -> str:
    cleaned = THINK_BLOCK_PATTERN.sub("", text)
    cleaned = REASONING_BLOCK_PATTERN.sub("", cleaned)
    return cleaned.strip()


def parse_reasoning_and_report_from_generation(generated_text: str) -> Tuple[str, str]:
    cleaned = clean_decoded_generation_text(generated_text)
    if not cleaned:
        return "", ""

    reasoning = _extract_reasoning_blocks(cleaned)

    if not reasoning:
        reasoning_marker = "Reasoning:"
        report_marker = "Report:"

        if reasoning_marker in cleaned:
            after_reasoning = cleaned.rsplit(reasoning_marker, 1)[-1].strip()
            if report_marker in after_reasoning:
                reasoning_part, report_part = after_reasoning.split(report_marker, 1)
                reasoning = reasoning_part.strip()
                report = _extract_report_body_from_text(report_part.strip())
                return reasoning, report

            reasoning = after_reasoning.strip()

    if not reasoning and "</think>" in cleaned and "<think>" not in cleaned:
        before, after = cleaned.split("</think>", 1)
        reasoning = before.strip()
        report = _extract_report_body_from_text(after.strip())
        return reasoning, report

    text_without_reasoning = _remove_reasoning_blocks(cleaned)
    report = _extract_report_body_from_text(text_without_reasoning)

    return reasoning, report


def extract_reasoning_from_generation(generated_text: str) -> str:
    reasoning, _ = parse_reasoning_and_report_from_generation(generated_text)
    return reasoning


def extract_report_from_generation(generated_text: str) -> str:
    _, report = parse_reasoning_and_report_from_generation(generated_text)
    return report

@dataclass(frozen=True)
class ReportMetricsBatch:
    bleu_1: List[float]
    bleu_2: List[float]
    bleu_3: List[float]
    bleu_4: List[float]
    rouge_l: List[float]
    meteor: List[float]
    cider: List[float]
    bleurt: List[float]
    ce_precision: List[float]
    ce_recall: List[float]
    ce_f1: List[float]


def compute_ce_precision_recall_f1(
    gt_report: str,
    generated_report: str,
    ignored_tokens: Sequence[str] = ("<MASK>",),
) -> Tuple[float, float, float]:
    """Compute CE Precision/Recall/F1 via normalized token-set overlap."""
    gt_tokens = set(
        normalize_tokens_for_scoring(
            tokens=tokenize_report_text(gt_report),
            ignored_tokens=ignored_tokens,
        )
    )
    pred_tokens = set(
        normalize_tokens_for_scoring(
            tokens=tokenize_report_text(generated_report),
            ignored_tokens=ignored_tokens,
        )
    )

    if not gt_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0

    true_positive = len(gt_tokens & pred_tokens)
    precision = float(true_positive) / float(len(pred_tokens)) if pred_tokens else 0.0
    recall = float(true_positive) / float(len(gt_tokens)) if gt_tokens else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
    return precision, recall, f1


class BleuSentenceScorer:

    def __init__(self) -> None:
        try:
            from sacrebleu.metrics import BLEU
        except Exception as exc:
            raise ImportError(
                "BLEU metrics require sacrebleu. Install with: pip install sacrebleu"
            ) from exc

        bleu_init_params = inspect.signature(BLEU.__init__).parameters
        bleu_kwargs: Dict[str, Any] = {"smooth_method": "exp"}

        if "effective_order" in bleu_init_params:
            bleu_kwargs["effective_order"] = True
        elif "use_effective_order" in bleu_init_params:
            bleu_kwargs["use_effective_order"] = True

        self._metrics = [
            BLEU(max_ngram_order=1, **bleu_kwargs),
            BLEU(max_ngram_order=2, **bleu_kwargs),
            BLEU(max_ngram_order=3, **bleu_kwargs),
            BLEU(max_ngram_order=4, **bleu_kwargs),
        ]

    def score_pair(self, reference: str, prediction: str) -> Tuple[float, float, float, float]:
        scores: List[float] = []
        for metric in self._metrics:
            result = metric.sentence_score(str(prediction), [str(reference)]).score
            scores.append(float(result) / 100.0)
        return float(scores[0]), float(scores[1]), float(scores[2]), float(scores[3])



def _lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0

    if len(tokens_a) < len(tokens_b):
        short = list(tokens_a)
        long = list(tokens_b)
    else:
        short = list(tokens_b)
        long = list(tokens_a)

    prev = [0] * (len(short) + 1)
    for token in long:
        curr = [0]
        for j, short_token in enumerate(short, start=1):
            if token == short_token:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return int(prev[-1])


class RougeLScorer:
    def __init__(self, use_stemmer: bool = True) -> None:
        self._scorer = None
        try:
            from rouge_score import rouge_scorer
        except Exception:
            return None

        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=bool(use_stemmer))

    def score_pair(self, reference: str, prediction: str) -> float:
        if self._scorer is not None:
            score = self._scorer.score(str(reference), str(prediction))["rougeL"].fmeasure
            return float(score)

        ref_tokens = tokenize_report_text(str(reference))
        pred_tokens = tokenize_report_text(str(prediction))

        lcs_len = _lcs_length(ref_tokens, pred_tokens)
        if lcs_len == 0:
            return 0.0

        precision = float(lcs_len) / float(max(1, len(pred_tokens)))
        recall = float(lcs_len) / float(max(1, len(ref_tokens)))
        denom = precision + recall
        return 2.0 * precision * recall / denom if denom > 0.0 else 0.0


class MeteorScorer:
    """METEOR scorer using NLTK."""

    def __init__(self) -> None:
        try:
            import nltk
            from nltk.translate.meteor_score import single_meteor_score
        except Exception as exc:
            raise ImportError("METEOR metrics require nltk. Install with: pip install nltk") from exc

        self._nltk = nltk
        self._meteor_fn = single_meteor_score
        self._ensure_resources()

    def _ensure_resources(self) -> None:
        corpora = ("wordnet", "omw-1.4")
        for corpus in corpora:
            try:
                self._nltk.data.find(f"corpora/{corpus}")
            except LookupError:
                self._nltk.download(corpus, quiet=True)

    def score_pair(self, reference: str, prediction: str) -> float:
        reference_tokens = tokenize_report_text(str(reference))
        prediction_tokens = tokenize_report_text(str(prediction))

        if not reference_tokens and not prediction_tokens:
            return 1.0
        if not reference_tokens or not prediction_tokens:
            return 0.0

        try:
            return float(self._meteor_fn(reference_tokens, prediction_tokens))
        except TypeError:
            return float(self._meteor_fn(str(reference), str(prediction)))

def estimate_cider_sigma(
    references: Sequence[str],
    minimum_sigma: float = 6.0,
    mean_length_ratio: float = 0.25,
) -> float:
    reference_lengths = [len(str(r).split()) for r in references if str(r).strip()]
    if not reference_lengths:
        return float(minimum_sigma)

    mean_length = float(np.mean(np.asarray(reference_lengths, dtype=np.float64)))
    return float(max(float(minimum_sigma), mean_length * float(mean_length_ratio)))


class CiderScorer:
    def __init__(self, n: int = 4, default_sigma: float = 6.0) -> None:
        try:
            from pycocoevalcap.cider.cider import Cider
        except Exception as exc:
            raise ImportError(
                "CIDEr metrics require pycocoevalcap. Install with: pip install pycocoevalcap"
            ) from exc

        self._cider_cls = Cider
        self._n = int(n)
        self._default_sigma = float(default_sigma)

    def score_corpus(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
        sigma: Optional[float] = None,
    ) -> List[float]:
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length.")

        effective_sigma = self._default_sigma if sigma is None else float(sigma)
        scorer = self._cider_cls(n=self._n, sigma=effective_sigma)

        gts = {str(i): [str(ref)] for i, ref in enumerate(references)}
        res = {str(i): [str(pred)] for i, pred in enumerate(predictions)}

        _, scores = scorer.compute_score(gts, res)
        return [float(x) for x in scores]



class BleurtScorer:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        batch_size: int = 16,
        max_length: int = 128,
    ) -> None:
        self._checkpoint = str(checkpoint)
        self._device = device
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._checkpoint,
            use_fast=True,
            trust_remote_code=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._checkpoint,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()

    @torch.inference_mode()
    def score_corpus(self, references: Sequence[str], predictions: Sequence[str]) -> List[float]:
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length.")

        scores: List[float] = []
        total = len(predictions)

        for start in range(0, total, self._batch_size):
            end = min(total, start + self._batch_size)
            ref_chunk = [str(x) for x in references[start:end]]
            pred_chunk = [str(x) for x in predictions[start:end]]

            encoded = self._tokenizer(
                ref_chunk,
                pred_chunk,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            outputs = self._model(**encoded)
            logits = outputs.logits.squeeze(-1).detach().to("cpu").numpy().tolist()
            scores.extend(float(x) for x in logits)

        return scores


class ReportMetricsComputer:
    def __init__(
        self,
        bleurt_checkpoint: str,
        bleurt_device: torch.device,
        bleurt_batch_size: int = 16,
        bleurt_max_length: int = 128,
    ) -> None:
        self._bleu = BleuSentenceScorer()
        self._rouge = RougeLScorer(use_stemmer=False)
        self._meteor = MeteorScorer()
        self._cider = CiderScorer()
        self._bleurt = BleurtScorer(
            checkpoint=bleurt_checkpoint,
            device=bleurt_device,
            batch_size=bleurt_batch_size,
            max_length=bleurt_max_length,
        )

    def compute(self, references: Sequence[str], predictions: Sequence[str]) -> ReportMetricsBatch:
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length.")

        bleu_1: List[float] = []
        bleu_2: List[float] = []
        bleu_3: List[float] = []
        bleu_4: List[float] = []
        rouge_l: List[float] = []
        meteor: List[float] = []
        ce_precision: List[float] = []
        ce_recall: List[float] = []
        ce_f1: List[float] = []

        for ref, pred in zip(references, predictions):
            b1, b2, b3, b4 = self._bleu.score_pair(reference=ref, prediction=pred)
            bleu_1.append(float(b1))
            bleu_2.append(float(b2))
            bleu_3.append(float(b3))
            bleu_4.append(float(b4))

            rouge_l.append(float(self._rouge.score_pair(reference=ref, prediction=pred)))
            meteor.append(float(self._meteor.score_pair(reference=ref, prediction=pred)))

            p, r, f1 = compute_ce_precision_recall_f1(gt_report=ref, generated_report=pred)
            ce_precision.append(float(p))
            ce_recall.append(float(r))
            ce_f1.append(float(f1))

        cider_sigma = estimate_cider_sigma(references=references, minimum_sigma=6.0, mean_length_ratio=0.25)
        cider = self._cider.score_corpus(references=references, predictions=predictions, sigma=cider_sigma)

        bleurt = self._bleurt.score_corpus(references=references, predictions=predictions)

        if len(cider) != len(predictions) or len(bleurt) != len(predictions):
            raise RuntimeError("Unexpected metric length mismatch.")

        return ReportMetricsBatch(
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            bleu_3=bleu_3,
            bleu_4=bleu_4,
            rouge_l=rouge_l,
            meteor=meteor,
            cider=[float(x) for x in cider],
            bleurt=[float(x) for x in bleurt],
            ce_precision=ce_precision,
            ce_recall=ce_recall,
            ce_f1=ce_f1,
        )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


@torch.no_grad()
def run_report_accuracy(
    model: EchoReportVlm,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder: ReportPromptBuilder,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_prompt_tokens: int,
    gen_max_new_tokens: int,
) -> float:
    model.eval()
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        video_features = batch["video_features"].to(device=device, dtype=autocast_dtype, non_blocking=True)
        video_mask = batch["video_mask"].to(device=device, non_blocking=True)
        gt_reports = list(batch["gt_report"])
        masked_reports = list(batch.get("masked_report", [""] * len(gt_reports)))
        study_ids = list(batch.get("exam_id", []))

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
        prompt_attention_mask = prompt_ids["attention_mask"].to(device=device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(device.type == "cuda"),
        ):
            generated = model.generate_report(
                tokenizer=tokenizer,
                video_features=video_features,
                video_mask=video_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                max_new_tokens=int(gen_max_new_tokens),
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                study_ids=study_ids if study_ids else None,
            )

        for gt_report, gen_text in zip(gt_reports, generated):
            predicted_report = extract_report_from_generation(gen_text)
            correct, total = compute_report_word_accuracy(
                gt_report=gt_report,
                predicted_report=predicted_report,
            )
            total_correct += int(correct)
            total_tokens += int(total)

    model.train()
    if total_tokens == 0:
        return 0.0
    return float(total_correct) / float(total_tokens)


@torch.no_grad()
def save_first_n_test_generations_to_csv(
    model: EchoReportVlm,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder: ReportPromptBuilder,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_prompt_tokens: int,
    gen_max_new_tokens: int,
    output_csv_path: Path,
    num_examples: int,
) -> int:
    model.eval()

    rows: List[Dict[str, str]] = []
    for batch in loader:
        exam_ids = list(batch["exam_id"])
        gt_reports = list(batch["gt_report"])
        masked_reports = list(batch.get("masked_report", [""] * len(exam_ids)))

        video_features = batch["video_features"].to(device=device, dtype=autocast_dtype, non_blocking=True)
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
        prompt_attention_mask = prompt_ids["attention_mask"].to(device=device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(device.type == "cuda"),
        ):
            generated_texts = model.generate_report(
                tokenizer=tokenizer,
                video_features=video_features,
                video_mask=video_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                max_new_tokens=int(gen_max_new_tokens),
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                study_ids=exam_ids if exam_ids else None,
            )

        for exam_id, gt_report, gen_text in zip(exam_ids, gt_reports, generated_texts):
            reasoning, generated_report = parse_reasoning_and_report_from_generation(gen_text)
            rows.append(
                {
                    "echo_id": str(exam_id),
                    "gt_report": str(gt_report),
                    "generated_report": str(generated_report),
                    "reasoning": str(reasoning),
                }
            )

            if len(rows) >= int(num_examples):
                break

        if len(rows) >= int(num_examples):
            break

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["echo_id", "gt_report", "generated_report", "reasoning"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    return int(len(rows))
