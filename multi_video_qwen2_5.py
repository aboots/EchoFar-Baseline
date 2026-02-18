from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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


def build_multi_video_single_prompt_messages(num_videos: int, prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "video"} for _ in range(num_videos)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_qwen2_5_vl_from_numpy_videos(
    model_name_or_path: str,
    videos_thw: Sequence[np.ndarray],
    prompt: str,
    fps: float | Sequence[float] = 1.0,
    max_new_tokens: int = 256,
) -> str:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    videos_thwc_rgb = [video_thw_to_thwc_rgb_uint8(v) for v in videos_thw]
    messages = build_multi_video_single_prompt_messages(num_videos=len(videos_thwc_rgb), prompt=prompt)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )

    inputs = processor(
        text=[text],
        videos=videos_thwc_rgb,
        fps=fps,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def demo() -> None:
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    t, h, w = 8, 256, 256
    video1 = (np.random.rand(t, h, w) * 255).astype(np.uint8)
    video2 = (np.random.rand(t, h, w) * 255).astype(np.uint8)

    prompt = "Compare Video 1 and Video 2. What is the main difference in motion?"
    output = run_qwen2_5_vl_from_numpy_videos(
        model_name_or_path=model_name_or_path,
        videos_thw=[video1, video2],
        prompt=prompt,
        fps=2.0,
        max_new_tokens=128,
    )
    print(output)


if __name__ == "__main__":
    demo()
