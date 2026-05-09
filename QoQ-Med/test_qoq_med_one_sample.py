import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from rclstream.datasets.private import echo
import numpy as np
import json
from pathlib import Path
from PIL import Image

# Paths
MODEL_ID = "ddvd233/QoQ-Med-VL-7B"

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
    print(f"Loading model {MODEL_ID}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading dataset...")
    patient_dataset = echo.EchoPatientDataset()
    
    # Pick one sample (exam_id 671)
    exam_id = "671"
    exam_id_to_idx = {str(row["exam_id"]): i for i, row in patient_dataset.patient_metadata.iterrows()}
    if exam_id not in exam_id_to_idx:
        print(f"Exam ID {exam_id} not found.")
        return
    
    idx = exam_id_to_idx[exam_id]
    sample = patient_dataset[idx]
    
    target_videos = sample["videos"][:1] # Just one video for testing
    v = target_videos[0]
    v_uint8 = video_thw_to_thwc_rgb_uint8(v)
    
    # Take 16 frames
    t = v_uint8.shape[0]
    indices = np.linspace(0, t - 1, 16, dtype=int)
    v_uint8 = v_uint8[indices]

    # Prepare message for QoQ-Med
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [Image.fromarray(f) for f in v_uint8],
                    "fps": 1.0,
                },
                {"type": "text", "text": "Generate an echocardiography findings report for this video."},
            ],
        }
    ]

    # Preparation for inference
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
    )
    inputs = inputs.to(model.device)

    # Inference
    print("Running inference...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("\nOutput:")
    print(output_text[0])

if __name__ == "__main__":
    main()
