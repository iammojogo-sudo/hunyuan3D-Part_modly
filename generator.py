import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import torch
from PIL import Image

# Assumes diffusers exposes HunyuanImagePipeline for HunyuanImage-2.1.
# If the actual class/module name differs, just fix the import below.
from diffusers import HunyuanImagePipeline


ProgressCallback = Optional[Callable[[float, str], None]]


class HunyuanImage21Generator:
    """
    Simple text-to-image generator around HunyuanImage-2.1 base model.

    Designed to be called by Modly with a params dict and optional
    progress / cancel hooks.
    """

    MODEL_ID = os.environ.get(
        "HUNYUANIMAGE_21_MODEL_ID",
        "Tencent-Hunyuan/HunyuanImage-2.1-base",
    )

    def __init__(self, model_dir: Optional[str] = None, outputs_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[HunyuanImagePipeline] = None

        self.model_dir = Path(model_dir) if model_dir else None
        self.outputs_dir = Path(outputs_dir) if outputs_dir else Path("./outputs")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        if self.pipe is not None:
            return

        self.pipe = HunyuanImagePipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.pipe.set_progress_bar_config(disable=True)

    def unload(self) -> None:
        self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate(
        self,
        params: Dict[str, Any],
        progress_cb: ProgressCallback = None,
        cancel_event: Optional[Any] = None,
    ) -> str:
        """
        Main entrypoint.

        params:
          - prompt: str
          - negative_prompt: Optional[str]
          - width: int
          - height: int
          - steps: int
          - guidance_scale: float
          - seed: int

        Returns:
          - path to generated PNG file (str)
        """
        self.load()
        assert self.pipe is not None

        prompt: str = params.get("prompt", "")
        negative_prompt: str = params.get("negative_prompt", "")
        width: int = int(params.get("width", 1024))
        height: int = int(params.get("height", 1024))
        steps: int = int(params.get("steps", 30))
        guidance_scale: float = float(params.get("guidance_scale", 5.0))
        seed: int = int(params.get("seed", 0))

        if not prompt:
            raise ValueError("prompt is required")

        if seed == 0:
            seed = random.randint(1, 2**31 - 1)

        if progress_cb:
            progress_cb(0.0, "starting HunyuanImage-2.1 generation")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if cancel_event is not None and getattr(cancel_event, "is_set", None):
            if cancel_event.is_set():
                raise RuntimeError("generation cancelled before start")

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

        out_name = f"hunyuanimage21_{seed}.png"
        out_path = self.outputs_dir / out_name
        image.save(out_path, format="PNG")

        if progress_cb:
            progress_cb(1.0, "done")

        return str(out_path)


# Optional module-level helper if Modly prefers a function entrypoint.
def generate(params: Dict[str, Any], progress_cb: ProgressCallback = None, cancel_event: Optional[Any] = None) -> str:
    gen = HunyuanImage21Generator()
    return gen.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)
