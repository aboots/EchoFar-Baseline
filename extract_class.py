#Anne's Note: this is only applied to our dataset
#there's also post-prcoess: like combine any moderate/serve or moderately/servely class
'''
Delete the concept classification name: LeftVentricle_wall_thickness, RightVentricle_wall_thickness,TricuspidValve_mobility,TricuspidValve_leaflet_thickening
Do not include/calculate the not well visualized / indeterminate/ determinate/ unknown/ not sure... 

(possible is wrong when you calulate the test classification matrices, also ignore the ground truth if in those labels!)


For the column names, change each row value (label) as following: 
AorticValve_regurgitation: combined moderate and serve to moderate
AorticValve_stenosis: combine moderate and serve to moderate
AorticValve_valve_thickening: combine moderate and serve to moderate
LeftAtrium_chamber_size: combine moderate_dilated and severe_dilated to moderate_dilated
LeftVentricle_chamber_size: combine moderate_dilated and severe_dilated to moderate_dilated
LeftVentricle_diastolic_function: combine moderate_diastolic_dysfunction and serve_diastolic_dysfunction to moderate_diastolic_dysfunction
LeftVentricle_systolic_function: combine moderately_depressed and severely_depressed to moderately_depressed
LeftVentricle_wall_motion: Combine severe_hypokinesis and mild_hypokinesis to abnormal 
MitralValve_leaflet_thickening: combined moderate and serve to moderate
MitralValve_regurgitation: combined moderate and serve to moderate
PulmonaryValveArtery_pulmonary_artery_systolic_pressure: delete the null
PulmonaryValveArtery_regurgitation: combined moderate and serve to moderate
RightAtrium_chamber_size:combine moderate_dilated and severe_dilated to moderate_dilated
RightVentricle_chamber_size: combine mild_dilated, moderate_dilated, severe_dilated to abnormal
RightVentricle_systolic_function: combined moderately_depressed and severely_depressed to moderately_depressed
TricuspidValve_regurgitation:combined moderate and serve to moderate
'''
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ConceptRule:
    concept_key: str
    c_label: str
    classifier: Callable[[str, random.Random], Optional[str]]


@dataclass(frozen=True)
class RegionSpec:
    canonical_name: str
    a_label: str
    prefix: str
    rules: Sequence[ConceptRule]


_NOT_WELL_VISUALIZED_PATTERNS = (
    r"\bnot well visualized\b",
    r"\bnot well seen\b",
    r"\bpoorly visualized\b",
    r"\bsuboptimally visualized\b",
    r"\blimited visualization\b",
    r"\bnot visualized\b",
)

_NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnone\b",
    r"\bnegative for\b",
)

_SEVERITY_CANONICALIZATION = {
    "mildly": "mild",
    "moderately": "moderate",
    "severely": "severe",
    "serve": "severe",
    "trace": "trivial",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract concept-level labels from echo results JSON and export a wide CSV "
            "with one row per study (echo_id) and one column per concept_name."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the results_*.json file from model inference.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to write the wide CSV file. If not provided, will use input filename with .csv extension.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to write extracted concept store JSON (optional).",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        help="Path to write concept_name/class_label summary counts (optional).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable the 'Section name: ...\nContent: ...' normalization.",
    )
    parser.add_argument(
        "--use-gt",
        action="store_true",
        help="Use ground_truth instead of prediction from the input JSON.",
    )
    parser.add_argument(
        "--missing-value",
        type=str,
        default="null",
        help="Value to write for missing concepts in the wide CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used when text contains ranges like 'mild to moderate' to "
            "choose one severity token."
        ),
    )
    parser.add_argument(
        "--no-print-summary",
        action="store_true",
        help="If set, do not print the summary counts to stdout.",
    )
    return parser


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


def parse_vlm_to_findings(text: str) -> Dict[str, str]:
    """
    Parses normalized text into a dictionary of findings per region.
    Example: 'Left Ventricle: Normal size. Right Ventricle: Normal function.'
    -> {'Left Ventricle': 'Normal size.', 'Right Ventricle': 'Normal function.'}
    """
    # This is a simple parser that looks for 'Region Name: ' at the start of lines
    # or after a newline.
    regions = [
        "Left Ventricle", "Right Ventricle", "Left Atrium", "Right Atrium",
        "Mitral Valve", "Tricuspid Valve", "Aortic Valve", "Pulmonary Valve/Artery",
        "Aorta", "Venous", "Pericardium/Other"
    ]
    
    findings = {}
    current_region = None
    current_text = []
    
    lines = text.split('\n')
    for line in lines:
        found_new_region = False
        for region in regions:
            if line.startswith(f"{region}:"):
                if current_region:
                    findings[current_region] = " ".join(current_text).strip()
                current_region = region
                current_text = [line[len(region)+1:].strip()]
                found_new_region = True
                break
        
        if not found_new_region and current_region:
            current_text.append(line.strip())
            
    if current_region:
        findings[current_region] = " ".join(current_text).strip()
        
    return findings


def load_json_records(input_path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSON or JSONL file.

    Args:
        input_path (Path): Path to the input file.

    Returns:
        List[Dict[str, Any]]: List of record dictionaries.
    """
    raw_text = input_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        data = json.loads(raw_text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                records.append(parsed)
        return records


def get_echo_id(record: Mapping[str, Any]) -> Any:
    echo_id = record.get("echo_id")
    if echo_id is not None:
        return echo_id
    return record.get("exam_id")


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = lowered.replace("\u2013", "-").replace("\u2014", "-")
    for source, target in _SEVERITY_CANONICALIZATION.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)

    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def split_into_sentences(text: str) -> List[str]:
    cleaned = text.replace("\n", " ").replace(";", ".")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in cleaned.split(".")]
    return [part for part in parts if part]


def contains_not_well_visualized(normalized_sentence: str) -> bool:
    return any(
        re.search(pattern, normalized_sentence)
        for pattern in _NOT_WELL_VISUALIZED_PATTERNS
    )


def detect_severity_token(
    normalized_sentence: str,
    rng: random.Random,
    allowed: Sequence[str],
) -> Optional[str]:
    range_match = re.search(
        r"\b(trivial|mild|moderate|severe)\s*(?:to|-|/)\s*(trivial|mild|moderate|severe)\b",
        normalized_sentence,
    )
    if range_match:
        left = range_match.group(1)
        right = range_match.group(2)
        candidates = [token for token in (left, right) if token in allowed]
        if candidates:
            return rng.choice(candidates)

    ordered = ["severe", "moderate", "mild", "trivial"]
    for token in ordered:
        if token in allowed and re.search(rf"\b{token}\b", normalized_sentence):
            return token

    return None


def is_negated(normalized_sentence: str, keyword_pattern: str) -> bool:
    negation_group = "|".join(_NEGATION_PATTERNS)
    pattern = rf"(?:{negation_group}).{{0,40}}(?:{keyword_pattern})"
    return re.search(pattern, normalized_sentence) is not None


def parse_percent_value(normalized_sentence: str) -> Optional[int]:
    match = re.search(
        r"\b(?:lvef|rvef|ef|ejection fraction)\b[^0-9]{0,15}(\d{1,3})\s*%?",
        normalized_sentence,
    )
    if not match:
        percent_match = re.search(r"\b(\d{1,3})\s*%\b", normalized_sentence)
        if percent_match:
            value = int(percent_match.group(1))
            if 0 <= value <= 100:
                return value
        return None

    value = int(match.group(1))
    if 0 <= value <= 100:
        return value
    return None


def classify_ef_value(percent: int) -> str:
    if percent >= 54:
        return "normal"
    if 41 <= percent <= 53:
        return "mildly_depressed"
    if 30 <= percent <= 40:
        return "moderately_depressed"
    return "severely_depressed"


def parse_diastolic_grade(normalized_sentence: str) -> Optional[int]:
    match = re.search(r"\bgrade\s*(i{1,3}|iv|[1-4])\b", normalized_sentence)
    if not match:
        return None

    token = match.group(1)
    if token.isdigit():
        return int(token)

    roman_to_int = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    return roman_to_int.get(token, None)


def classify_chamber_size(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    has_size_context = any(
        token in text
        for token in (
            "size",
            "volume",
            "cavity",
            "dimension",
            "diameter",
            "by linear dimension",
        )
    )
    has_dilation_context = any(
        token in text for token in ("dilat", "enlarg", "increased size", "increased volume")
    )

    if not (has_size_context or has_dilation_context):
        return None

    if "normal size" in text or ("normal" in text and "size" in text and not has_dilation_context):
        return "normal"

    if is_negated(text, r"\b(dilat\w*|enlarg\w*)\b"):
        return "normal"

    if (
        re.search(r"\b(dilat\w*|enlarg\w*)\b", text)
        or "increased size" in text
        or "increased volume" in text
    ):
        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity is None:
            severity = "mild"
        return f"{severity}_dilated"

    if "small" in text and "size" in text:
        return "small"

    return None


def classify_wall_thickness(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if "concentric remodeling" in text or "concentric remodelling" in text:
        return "concentric_remodeling"

    if contains_not_well_visualized(text):
        return "not_well_visualized"

    thickness_context = any(
        token in text for token in ("wall thickness", "thickness", "hypertrophy", "lvh", "rvh")
    )
    if not thickness_context and "concentric" not in text:
        return None

    if is_negated(text, r"\b(hypertrophy|lvh|rvh)\b"):
        return "normal"

    if "normal wall thickness" in text or ("normal" in text and "thickness" in text):
        return "normal"

    if (
        re.search(r"\b(hypertrophy|lvh|rvh)\b", text)
        or "increased wall thickness" in text
        or ("increased" in text and "thickness" in text)
    ):
        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity is None:
            severity = "mild"
        return f"{severity}ly_increased"

    return None


def classify_systolic_function(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    has_function_context = any(
        token in text
        for token in (
            "systolic",
            "ejection fraction",
            "lvef",
            "rvef",
            "ef",
            "contractile",
            "contractility",
            "function",
        )
    )
    if not has_function_context or "diastolic" in text:
        return None

    if "hyperdynamic" in text:
        return "normal"

    percent = parse_percent_value(text)
    if percent is not None:
        return classify_ef_value(percent)

    if re.search(r"\b(normal|preserved|intact)\b", text) and (
        "systolic" in text or "function" in text or "ef" in text
    ):
        return "normal"

    if re.search(r"\b(reduced|depressed|decrease|decreased|diminished|impaired)\b", text):
        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity is None:
            severity = "mild"
        return f"{severity}ly_depressed"

    return None


def classify_diastolic_function(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    has_context = (
        "diastolic" in text
        or "impaired relaxation" in text
        or "diastolic dysfunction" in text
        or "grade" in text
    )
    if not has_context:
        return None

    if "normal diastolic function" in text:
        return "normal"

    if "indeterminate" in text or "unable to assess" in text or "cannot assess" in text:
        return "indeterminate"

    if "impaired relaxation" in text:
        return "mild_diastolic_dysfunction"

    if "diastolic dysfunction" in text or "diastolic" in text:
        grade = parse_diastolic_grade(text)
        if grade == 1:
            return "mild_diastolic_dysfunction"
        if grade == 2:
            return "moderate_diastolic_dysfunction"
        if grade is not None and grade >= 3:
            return "severe_diastolic_dysfunction"

        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity == "mild":
            return "mild_diastolic_dysfunction"
        if severity == "moderate":
            return "moderate_diastolic_dysfunction"
        if severity == "severe":
            return "severe_diastolic_dysfunction"

        return "mild_diastolic_dysfunction"

    return None


def classify_wall_motion(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    has_motion_context = (
        "wall motion" in text
        or "rwma" in text
        or re.search(r"\b(hypokines\w*|akines\w*|dyskines\w*)\b", text) is not None
    )
    if not has_motion_context:
        return None

    if (
        "no regional wall motion abnormal" in text
        or "normal wall motion" in text
        or is_negated(text, r"\bwall motion abnormal\w*\b")
    ):
        return "normal"

    if re.search(r"\b(hypokines\w*|akines\w*|dyskines\w*)\b", text) or "wall motion abnormal" in text:
        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity is None:
            severity = "mild"

        if "akines" in text:
            base = "akinesis"
        elif "dyskines" in text:
            base = "dyskinesis"
        else:
            base = "hypokinesis"

        return f"{severity}_{base}"

    return None


def classify_filling_pressure(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    has_context = "filling pressure" in text or "filling pressures" in text or "e/e" in text
    if not has_context:
        return None

    if contains_not_well_visualized(text):
        return "not_well_visualized"

    if "indeterminate" in text:
        return "indeterminate"

    if re.search(r"\b(normal|wnl|within normal limits)\b", text):
        return "normal"

    if re.search(r"\b(elevated|increase|increased|high)\b", text):
        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity is None:
            severity = "mild"
        return f"{severity}ly_elevated"

    return None


def classify_valve_leaflet_thickening(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    relevant = any(
        token in text
        for token in (
            "leaflet",
            "leaflets",
            "thick",
            "thickening",
            "sclerosis",
            "normal structure",
            "normal valve",
        )
    )
    if not relevant:
        return None

    if is_negated(text, r"\bthick\w*\b"):
        return "normal"

    if (
        "normal valve leaflets" in text
        or "normal leaflets" in text
        or "normal structure" in text
        or "normal valve" in text
    ):
        return "normal"

    if re.search(r"\b(thick\w*|sclerosis)\b", text):
        severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
        if severity is None:
            severity = "mild"
        return severity

    return None


def classify_valve_regurgitation(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)

    if contains_not_well_visualized(text):
        return "not_well_visualized"

    mentions_regurg = "regurg" in text or "insufficien" in text
    if not mentions_regurg:
        return None

    if is_negated(text, r"\b(regurg\w*|insufficien\w*)\b") or "no evidence of regurg" in text:
        return "normal"

    if "insufficient tricuspid regurgitation" in text and "estimat" in text:
        return "indeterminate"

    severity = detect_severity_token(text, rng, allowed=("trivial", "mild", "moderate", "severe"))
    if severity is None:
        if "trace" in sentence.lower():
            return "trivial"
        return "mild"

    return severity


def classify_valve_mobility(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    if "mobility" not in text and "mobile" not in text and "restricted" not in text:
        return None

    if re.search(r"\b(normal|preserved)\b", text) and ("mobility" in text or "mobile" in text):
        return "normal"

    if "restricted" in text or "reduced mobility" in text:
        return "restricted"

    return None


def classify_leaflet_calcification(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    if "calcif" not in text and "calcified" not in text:
        return None

    if "annular" in text or "annulus" in text or re.search(r"\bmac\b", text):
        return None

    if is_negated(text, r"\bcalcif\w*\b"):
        return "normal"

    severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
    return severity or "mild"


def classify_annular_calcification(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    has_context = (
        "annular calcification" in text
        or "annulus calcification" in text
        or re.search(r"\bmac\b", text) is not None
    )
    if not has_context:
        return None

    if is_negated(text, r"\b(annular calcification|annulus calcification|mac)\b"):
        return "normal"

    severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
    return severity or "mild"


def classify_aortic_stenosis(sentence: str, rng: random.Random) -> Optional[str]:
    text = normalize_text(sentence)
    if contains_not_well_visualized(text):
        return "not_well_visualized"

    has_stenosis_context = "stenosis" in text or "aortic stenosis" in text or "as " in text
    if not has_stenosis_context:
        if "normal structure" in text or "normal valve" in text:
            return "normal"
        return None

    if is_negated(text, r"\bstenosis\b"):
        return "normal"

    severity = detect_severity_token(text, rng, allowed=("mild", "moderate", "severe"))
    return severity or "mild"


def build_region_specs() -> List[RegionSpec]:
    left_ventricle = RegionSpec(
        canonical_name="Left Ventricle",
        a_label="A1",
        prefix="LeftVentricle",
        rules=(
            ConceptRule("chamber_size", "C1", classify_chamber_size),
            ConceptRule("wall_thickness", "C2", classify_wall_thickness),
            ConceptRule("systolic_function", "C3", classify_systolic_function),
            ConceptRule("diastolic_function", "C4", classify_diastolic_function),
            ConceptRule("wall_motion", "C5", classify_wall_motion),
            ConceptRule("filling_pressure", "C6", classify_filling_pressure),
        ),
    )

    right_ventricle = RegionSpec(
        canonical_name="Right Ventricle",
        a_label="A2",
        prefix="RightVentricle",
        rules=(
            ConceptRule("chamber_size", "C1", classify_chamber_size),
            ConceptRule("wall_thickness", "C2", classify_wall_thickness),
            ConceptRule("systolic_function", "C3", classify_systolic_function),
        ),
    )

    right_atrium = RegionSpec(
        canonical_name="Right Atrium",
        a_label="A3",
        prefix="RightAtrium",
        rules=(
            ConceptRule("chamber_size", "C1", classify_chamber_size),
            ConceptRule("filling_pressure", "C6", classify_filling_pressure),
        ),
    )

    left_atrium = RegionSpec(
        canonical_name="Left Atrium",
        a_label="A4",
        prefix="LeftAtrium",
        rules=(ConceptRule("chamber_size", "C1", classify_chamber_size),),
    )

    mitral_valve = RegionSpec(
        canonical_name="Mitral Valve",
        a_label="A5",
        prefix="MitralValve",
        rules=(
            ConceptRule("leaflet_calcification", "C10", classify_leaflet_calcification),
            ConceptRule("annular_calcification", "C12", classify_annular_calcification),
            ConceptRule("leaflet_thickening", "C7", classify_valve_leaflet_thickening),
            ConceptRule("regurgitation", "C8", classify_valve_regurgitation),
        ),
    )

    tricuspid_valve = RegionSpec(
        canonical_name="Tricuspid Valve",
        a_label="A6",
        prefix="TricuspidValve",
        rules=(
            ConceptRule("leaflet_thickening", "C7", classify_valve_leaflet_thickening),
            ConceptRule("regurgitation", "C8", classify_valve_regurgitation),
            ConceptRule("mobility", "C9", classify_valve_mobility),
        ),
    )

    aortic_valve = RegionSpec(
        canonical_name="Aortic Valve",
        a_label="A7",
        prefix="AorticValve",
        rules=(
            ConceptRule("valve_thickening", "C7", classify_valve_leaflet_thickening),
            ConceptRule("stenosis", "C11", classify_aortic_stenosis),
            ConceptRule("regurgitation", "C8", classify_valve_regurgitation),
        ),
    )

    pulmonary_valve = RegionSpec(
        canonical_name="Pulmonary Valve/Artery",
        a_label="A8",
        prefix="PulmonaryValveArtery",
        rules=(
            ConceptRule("regurgitation", "C8", classify_valve_regurgitation),
        ),
    )

    return [
        left_ventricle,
        right_ventricle,
        left_atrium,
        right_atrium,
        mitral_valve,
        tricuspid_valve,
        aortic_valve,
        pulmonary_valve,
    ]


def build_region_aliases() -> Dict[str, List[str]]:
    return {
        "Left Ventricle": ["LV", "Left ventricle"],
        "Right Ventricle": ["RV", "Right ventricle"],
        "Left Atrium": ["LA", "Left atrium"],
        "Right Atrium": ["RA", "Right atrium"],
        "Mitral Valve": ["Mitral valve", "MV"],
        "Tricuspid Valve": ["Tricuspid valve", "TV"],
        "Aortic Valve": ["Aortic valve", "AV"],
        "Pulmonary Valve/Artery": ["Pulmonary valve", "PV", "Pulmonary artery", "PA"],
    }


def find_region_text(findings: Mapping[str, Any], canonical_region_name: str) -> Optional[str]:
    if canonical_region_name in findings:
        value = findings.get(canonical_region_name)
        return str(value) if value is not None else None

    normalized_target = normalize_text(canonical_region_name)
    for key, value in findings.items():
        if normalize_text(str(key)) == normalized_target:
            return str(value) if value is not None else None

    aliases = build_region_aliases().get(canonical_region_name, [])
    normalized_aliases = {normalize_text(alias) for alias in aliases}
    for key, value in findings.items():
        if normalize_text(str(key)) in normalized_aliases:
            return str(value) if value is not None else None

    return None


def extract_concepts_for_region(
    region_spec: RegionSpec,
    region_text: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    sentences = split_into_sentences(region_text)
    extracted: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(sentences, start=1):
        for rule in region_spec.rules:
            class_label = rule.classifier(sentence, rng)
            if class_label is None:
                continue

            concept_name = f"{region_spec.prefix}_{rule.concept_key}"
            extracted.append(
                {
                    "anatomical_region": region_spec.canonical_name,
                    "a_label": region_spec.a_label,
                    "c_label": rule.c_label,
                    "concept_key": rule.concept_key,
                    "concept_name": concept_name,
                    "class_label": class_label,
                    "sentence_index": sentence_index,
                    "sentence": sentence.strip(),
                }
            )

    return extracted


def process_records(
    records: Sequence[Mapping[str, Any]],
    region_specs: Sequence[RegionSpec],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    processed: List[Dict[str, Any]] = []

    for record in records:
        echo_id = get_echo_id(record)
        findings = record.get("findings", {})
        findings_dict = findings if isinstance(findings, dict) else {}

        record_concepts: List[Dict[str, Any]] = []
        for region_spec in region_specs:
            region_text = find_region_text(findings_dict, region_spec.canonical_name)
            if not region_text:
                continue
            record_concepts.extend(extract_concepts_for_region(region_spec, region_text, rng))

        processed.append(
            {
                "echo_id": echo_id,
                "concepts": record_concepts,
            }
        )

    return processed


def write_json(output_path: Path, data: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_expected_concept_names(region_specs: Sequence[RegionSpec]) -> List[str]:
    concept_names: List[str] = []
    for region_spec in region_specs:
        for rule in region_spec.rules:
            concept_names.append(f"{region_spec.prefix}_{rule.concept_key}")
    return sorted(concept_names)


def build_observed_concept_names(extracted_records: Sequence[Mapping[str, Any]]) -> Set[str]:
    observed: Set[str] = set()
    for record in extracted_records:
        concepts = record.get("concepts", [])
        if not isinstance(concepts, list):
            continue
        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            concept_name = concept.get("concept_name")
            if concept_name:
                observed.add(str(concept_name))
    return observed


def format_stream_id(stream_id: Any, missing_value: str) -> str:
    if stream_id is None:
        return missing_value

    if isinstance(stream_id, (list, dict)):
        return json.dumps(stream_id, ensure_ascii=False)

    return str(stream_id)


def choose_label_for_concept(entries: Sequence[Tuple[str, int]]) -> str:
    """
      1) Most frequent class_label (mode) ////
      2) Tie-breaker: earliest sentence_index
      3) Tie-breaker: lexicographic order for determinism
    """
    label_counts = Counter(label for label, _ in entries)
    max_count = max(label_counts.values())
    candidate_labels = {label for label, count in label_counts.items() if count == max_count}

    if len(candidate_labels) == 1:
        return next(iter(candidate_labels))

    earliest_by_label: Dict[str, int] = {}
    for label, sentence_index in entries:
        if label not in candidate_labels:
            continue
        existing = earliest_by_label.get(label)
        if existing is None or sentence_index < existing:
            earliest_by_label[label] = sentence_index

    ordered_candidates = sorted(
        candidate_labels,
        key=lambda lbl: (earliest_by_label.get(lbl, 10**9), lbl),
    )
    return ordered_candidates[0]


def resolve_record_concepts(concepts: Any) -> Dict[str, str]:
    if not isinstance(concepts, list):
        return {}

    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        concept_name = concept.get("concept_name")
        class_label = concept.get("class_label")
        if not concept_name or class_label is None:
            continue

        sentence_index_value = concept.get("sentence_index")
        sentence_index = int(sentence_index_value) if isinstance(sentence_index_value, int) else 10**9

        grouped[str(concept_name)].append((str(class_label), sentence_index))

    resolved: Dict[str, str] = {}
    for concept_name, entries in grouped.items():
        resolved[concept_name] = choose_label_for_concept(entries)

    return resolved


def build_wide_rows_and_distribution(
    extracted_records: Sequence[Mapping[str, Any]],
    concept_columns: Sequence[str],
    missing_value: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    combination_to_echo_ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    wide_rows: List[Dict[str, str]] = []

    for record in extracted_records:
        echo_id_value = get_echo_id(record)
        if echo_id_value is None:
            echo_id_str = missing_value
        else:
            echo_id_str = str(echo_id_value)

        resolved_concepts = resolve_record_concepts(record.get("concepts"))

        row: Dict[str, str] = {
            "echo_id": echo_id_str,
        }

        for concept_name in concept_columns:
            class_label = resolved_concepts.get(concept_name, missing_value)
            row[concept_name] = class_label
            if echo_id_value is not None:
                combination_to_echo_ids[(concept_name, class_label)].add(echo_id_str)

        wide_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (concept_name, class_label), echo_ids in combination_to_echo_ids.items():
        summary_rows.append(
            {
                "concept_name": concept_name,
                "class_label": class_label,
                "echo_id_count": len(echo_ids),
            }
        )

    summary_rows.sort(key=lambda r: (r["concept_name"], r["class_label"]))
    return wide_rows, summary_rows


def write_wide_csv(
    output_path: Path,
    rows: Sequence[Mapping[str, str]],
    concept_columns: Sequence[str],
) -> None:
    fieldnames = ["echo_id", *concept_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_summary_csv(output_path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["concept_name", "class_label", "echo_id_count"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(dict(row))


def print_summary(summary_rows: Sequence[Mapping[str, Any]]) -> None:
    concept_to_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        concept_to_rows[str(row["concept_name"])].append(row)

    for concept_name in sorted(concept_to_rows.keys()):
        rows = concept_to_rows[concept_name]
        sorted_rows = sorted(
            rows,
            key=lambda r: (-int(r["echo_id_count"]), str(r["class_label"])),
        )
        print(concept_name)
        for row in sorted_rows:
            class_label = str(row["class_label"])
            echo_id_count = int(row["echo_id_count"])
            print(f"  {class_label}: {echo_id_count}")
        print("")


def detect_input_kind(records: Sequence[Mapping[str, Any]]) -> str:
    for record in records:
        if not isinstance(record, dict):
            continue
        if "findings" in record:
            return "findings"
        if "concepts" in record:
            return "extracted"
    raise ValueError("Could not detect input kind; expected 'findings' or 'concepts' in records.")


def run(
    input_path: Path,
    output_json_path: Optional[Path],
    output_csv_path: Path,
    output_summary_csv_path: Optional[Path],
    missing_value: str,
    seed: int,
    should_print_summary: bool,
    no_normalize: bool,
    use_gt: bool,
) -> None:
    """
        input_path (Path): Input JSON/JSONL path.
        output_json_path (Path): Output extracted JSON path.
        output_csv_path (Path): Output wide CSV path.
        output_summary_csv_path (Path): Output summary CSV path.
        missing_value (str): Placeholder value for missing concept labels.
        seed (int): Random seed used for severity ranges.
        should_print_summary (bool): Whether to print summary to stdout.
        no_normalize (bool): Disable VLM normalization.
        use_gt (bool): Use ground_truth field.
    """
    records = load_json_records(input_path)
    if not records:
        raise SystemExit(f"No records found in {input_path}")

    region_specs = build_region_specs()
    
    # Process each record to extract findings
    processed_records = []
    for record in records:
        text = record.get("ground_truth" if use_gt else "prediction", "")
        if not no_normalize:
            text = normalize_vlm_prediction(text)
        
        findings = parse_vlm_to_findings(text)
        processed_records.append({
            "echo_id": get_echo_id(record),
            "findings": findings
        })

    extracted_records = process_records(processed_records, region_specs=region_specs, seed=seed)
    
    if output_json_path:
        write_json(output_json_path, extracted_records)

    expected_concepts = set(build_expected_concept_names(region_specs))
    observed_concepts = build_observed_concept_names(extracted_records)
    concept_columns = sorted(expected_concepts | observed_concepts)

    wide_rows, summary_rows = build_wide_rows_and_distribution(
        extracted_records=extracted_records,
        concept_columns=concept_columns,
        missing_value=missing_value,
    )

    write_wide_csv(output_csv_path, rows=wide_rows, concept_columns=concept_columns)
    
    if output_summary_csv_path:
        write_summary_csv(output_summary_csv_path, summary_rows=summary_rows)

    if should_print_summary:
        print_summary(summary_rows)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_csv = Path(args.output_csv) if args.output_csv else input_path.with_suffix(".csv")
    output_json = Path(args.output_json) if args.output_json else None
    output_summary = Path(args.output_summary_csv) if args.output_summary_csv else None

    run(
        input_path=input_path,
        output_json_path=output_json,
        output_csv_path=output_csv,
        output_summary_csv_path=output_summary,
        missing_value=str(args.missing_value),
        seed=int(args.seed),
        should_print_summary=not bool(args.no_print_summary),
        no_normalize=args.no_normalize,
        use_gt=args.use_gt,
    )


if __name__ == "__main__":
    main()
