"""
Supported example model ids:
- Qwen/Qwen2-VL-7B-Instruct
- Qwen/Qwen2-VL-72B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct
- Qwen/Qwen2.5-VL-72B-Instruct
- Qwen/Qwen3-VL-4B-Instruct
- Qwen/Qwen3-VL-8B-Instruct
- Qwen/Qwen3-VL-32B-Instruct
- google/gemma-3-12b-it
"""


#!/usr/bin/env python3
# vlm_loader.py
# Switchable loader for: Qwen2-VL, Qwen2.5-VL, Gemma 3, Qwen2.5-Omni, Qwen3-Omni, Qwen3-VL

import os
import torch
from transformers import AutoProcessor


def _maybe_login_hf() -> None:
    """Login to Hugging Face if HF_TOKEN is provided."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return
    from huggingface_hub import login
    login(token=token)


def load_model_and_processor(model_id: str):
    """
    Return (model, processor) for the given model_id.

    Supported families:
      - Qwen2-VL:     "Qwen/Qwen2-VL-*-Instruct"
      - Qwen2.5-VL:   "Qwen/Qwen2.5-VL-*-Instruct"
      - Qwen3-VL:     "Qwen/Qwen3-VL-*-Instruct" / "Qwen/Qwen3-VL-*-Thinking"
      - Gemma 3:      "google/gemma-3-*"
      - Qwen2.5-Omni: "Qwen/Qwen2.5-Omni-*"
      - Qwen3-Omni:   "Qwen/Qwen3-Omni-*"
    """
    _maybe_login_hf()

    model_name = model_id

    if model_id.startswith("Qwen/Qwen2-VL-"):
        # Qwen2-VL vision-language model
        # Class: Qwen2VLForConditionalGeneration
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        # AutoProcessor will instantiate Qwen2VLProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    elif model_id.startswith("Qwen/Qwen2.5-VL-"):
        # Qwen2.5-VL vision-language model
        # Class: Qwen2_5_VLForConditionalGeneration
        # Note: Transformers may require recent versions (often source install).
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        # AutoProcessor should instantiate the right processor for qwen2_5_vl
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    elif model_id.startswith("Qwen/Qwen3-VL-"):
        processor = AutoProcessor.from_pretrained(model_name)

        if ("-A3B" in model_id) or ("-A22B" in model_id):
            from transformers import Qwen3VLMoeForConditionalGeneration
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        else:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

        return model, processor

    elif model_id.startswith("google/gemma-3-"):
        # Gemma 3 (vision-language variants)
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    elif model_id.startswith("Qwen/Qwen2.5-Omni"):
        # Qwen2.5-Omni: dedicated model/processor classes
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        return model, processor

    elif model_id.startswith("Qwen/Qwen3-Omni"):
        # Qwen3-Omni: Thinker-only model for text output
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor

        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        return model, processor

    elif model_id.startswith("llava-hf/llava-"):
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    raise NotImplementedError(f"not implemented: {model_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_id)
    print("Loaded:", type(model).__name__, "|", type(processor).__name__)
