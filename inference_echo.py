import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from rclstream.datasets.private import echo
import numpy as np
from typing import List, Dict, Any, Sequence
import json
from pathlib import Path

# Paths
GT_JSON_PATH = Path("/home/mahdi.abootorabi/EchoFAR/findings_token_all.json")
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

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

    # Load dataset
    print("Loading EchoPatientDataset...")
    patient_dataset = echo.EchoPatientDataset()
    
    # ICL Prompt Construction
    system_prompt = (
        "You are a medical assistant generating an echocardiography findings report.\n"
        "Write one section per line in the format 'Section: content'.\n"
    )

    example1 = (
        "Example 1:\n"
        "Left Ventricle: Normal size based on linear index dimension. Normal ejection fraction. No regional wall motion abnormalities. Septal flattening of the left ventricle due to volume loaded right ventricle. Indeterminate diastolic function. Indeterminate filling pressure.\n"
        "Right Ventricle: Dilated by linear dimension. Mildly decreased systolic function.\n"
        "Left Atrium: Severely dilated by biplane volume index.\n"
        "Right Atrium: Dilated by volume index. Dilated coronary sinus.\n"
        "Mitral Valve: Mechanical prosthesis. Prosthesis well seated. Mean pressure gradient: 9 mmHg at HR of 93 bpm. Trivial regurgitation. Regurgitation not well seen due to shadowing, may be underestimated.\n"
        "Tricuspid Valve: Malcoapting leaflets. Normal mobility of the leaflets. Severe regurgitation.\n"
        "Aortic Valve: Mechanical prosthesis. Prosthesis well seated.\n"
        "Pulmonary Valve/Artery: Not well seen. Mild regurgitation. RVSP/PASP may be underestimated due to severe tricuspid regurgitation.\n"
        "Aorta: Normal sinuses of Valsalva by index dimension. Normal sinuses of Valsalva by linear dimension. Normal proximal ascending aorta by index dimension. Normal proximal ascending aorta by linear dimension.\n"
        "Venous: Hepatic vein has a systolic flow reversal, suggestive of significant tricuspid regurgitation. Inferior vena cava is dilated (>21 mm) with less than 50% respiratory variation.\n"
        "Pericardium/Other: Trivial pericardial effusion. Left and right pleural effusion.\n"
    )

    example2 = (
        "Example 2:\n"
        "Left Ventricle: Normal LV size based on indexed linear dimension. Normal systolic function. Ejection fraction by visual estimation is 60 %.\n"
        "Right Ventricle: Normal size by linear measurement. Normal systolic function.\n"
        "Left Atrium: Normal in size by volume index.\n"
        "Right Atrium: Normal in size by volume index.\n"
        "Mitral Valve: Normal valve leaflets. Trivial regurgitation.\n"
        "Tricuspid Valve: Normal valve leaflets. Normal mobility of the leaflets. Trivial regurgitation.\n"
        "Aortic Valve: Trileaflet valve; normal structure. No evidence of valvular regurgitation.\n"
        "Pulmonary Valve/Artery: Normal valve. Mild regurgitation.\n"
        "Aorta: Normal proximal ascending aorta by index dimension.\n"
    )

    prompt_body = "Generate an echocardiography findings report for the provided 5 videos, following the format above."

    # Full context prompt
    full_icl_prompt = f"{system_prompt}\n{example1}\n{example2}\n{prompt_body}"

    # Take one patient as a test
    idx = 0 
    sample = patient_dataset[idx]
    exam_id = sample["exam_id"]
    videos = sample["videos"] # List of (T, H, W)
    
    # Take first 5 videos
    selected_videos = videos[:5]
    num_videos = len(selected_videos)
    print(f"Processing exam_id: {exam_id} with {num_videos} videos.")

    # Convert to RGB (T, H, W, 3)
    videos_rgb = [video_thw_to_thwc_rgb_uint8(v) for v in selected_videos]

    # Build messages
    content = []
    for _ in range(num_videos):
        content.append({"type": "video"})
    content.append({"type": "text", "text": full_icl_prompt})

    messages = [
        {"role": "user", "content": content}
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Note: Using processor directly with videos like in Qwen2.5-VL
    inputs = processor(
        text=[text],
        videos=videos_rgb,
        fps=1.0, # Adjust if needed
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Inference
    print("Generating report...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n--- Generated Report ---")
    print(output_text[0])
    print("------------------------")

if __name__ == "__main__":
    main()
