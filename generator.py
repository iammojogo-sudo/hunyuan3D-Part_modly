import os
import random
from typing import Any, Dict

import torch
from PIL import Image

# Assumes diffusers has HunyuanImagePipeline for HunyuanImage-2.1
# If the class name/module differs, just fix the import below.
from diffusers import HunyuanImagePipeline


class HunyuanImage21Generator:
    """
    Modly generator wrapper for Tencent HunyuanImage-2.1 (base-only, text-to-image).
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None

    def _load_pipeline(self) -> None:
        if self.pipe is not None:
            return

        model_id = os.environ.get(
            "HUNYUANIMAGE_21_MODEL_ID",
            "Tencent-Hunyuan/HunyuanImage-2.1-base",
        )

        self.pipe = HunyuanImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # Small perf tweaks
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modly entrypoint.

        Expected inputs (from manifest.json):
          - prompt: str
          - negative_prompt: Optional[str]
          - width: int
          - height: int
          - steps: int
          - guidance_scale: float
          - seed: int
        """
        self._load_pipeline()

        prompt: str = inputs.get("prompt", "")
        negative_prompt: str = inputs.get("negative_prompt", "")
        width: int = int(inputs.get("width", 1024))
        height: int = int(inputs.get("height", 1024))
        steps: int = int(inputs.get("steps", 30))
        guidance_scale: float = float(inputs.get("guidance_scale", 5.0))
        seed: int = int(inputs.get("seed", 0))

        if not prompt:
            raise ValueError("prompt is required")

        if seed == 0:
            seed = random.randint(1, 2**31 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        image: Image.Image = out.images[0]

        return {
            "image": image,
            "meta": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "model": "HunyuanImage-2.1-base",
            },
        }


# Modly usually expects a factory or a module-level symbol.
# If your Modly integration uses a different hook, just point it to this class.

def load() -> HunyuanImage21Generator:
    return HunyuanImage21Generator()
