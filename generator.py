# generator.py
"""
Single top-level Generator for Modly.
Modly will import Generator from this file (manifest.entry -> generator.py, generator_class -> "Generator").
This generator expects the HF repo to be downloaded by setup.py into:
  <extension_root>/models/hunyuan_t2i_turbo_modly/
"""

import os
import io
import base64
from typing import Optional, Dict, Any
from PIL import Image

ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "models", "hunyuan_t2i_turbo_modly")
MODEL_DIR = os.path.normpath(MODEL_DIR)

_pipeline_cache = None

def _load_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    import torch
    from diffusers import DiffusionPipeline

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model not found at {MODEL_DIR}. Run the extension Install/Download first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None
    )
    pipeline = pipeline.to(device)

    _pipeline_cache = {"pipeline": pipeline, "torch": torch, "device": device}
    return _pipeline_cache

class Generator:
    """
    Exposed class name: Generator
    Modly will call into this class according to its generator protocol.
    """

    def __init__(self):
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            self._pipe = _load_pipeline()
        return self._pipe

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        steps: int = 28,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1
    ) -> Dict[str, Any]:
        info = self._ensure()
        pipeline = info["pipeline"]
        torch = info["torch"]
        device = info["device"]

        width = max(64, min(2048, int(width)))
        height = max(64, min(2048, int(height)))
        steps = max(1, min(200, int(steps)))
        guidance_scale = float(guidance_scale)

        generator = None
        if seed is not None and int(seed) >= 0:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )

        images = getattr(result, "images", None)
        if images is None:
            images = result if isinstance(result, list) else []

        if not images:
            raise RuntimeError("Pipeline returned no images")

        img = images[0]
        if not isinstance(img, Image.Image):
            try:
                import numpy as _np
                arr = img.cpu().permute(1, 2, 0).numpy()
                arr = (_np.clip(arr * 255, 0, 255)).astype("uint8")
                img = Image.fromarray(arr)
            except Exception:
                raise RuntimeError("Unable to convert pipeline output to PIL Image")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image": encoded}
