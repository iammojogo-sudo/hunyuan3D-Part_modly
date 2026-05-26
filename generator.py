# models/hunyuan_t2i_turbo/generator.py
import os
import io
import base64
from typing import Optional, Dict, Any
from PIL import Image

MODEL_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "hunyuan_t2i_turbo")
MODEL_LOCAL_DIR = os.path.normpath(MODEL_LOCAL_DIR)

_pipeline_cache = None

def _load_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    import torch
    from diffusers import DiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_dir = MODEL_LOCAL_DIR
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"HunyuanDiT-Turbo model not found at {model_dir}. Run setup install first.")

    pipeline = DiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None
    )

    pipeline = pipeline.to(device)

    _pipeline_cache = {
        "pipeline": pipeline,
        "device": device,
        "torch": torch
    }
    return _pipeline_cache

class Generator:
    """
    Modly-compatible generator class.
    Must be importable as models.hunyuan_t2i_turbo.generator.Generator
    and expose a run(...) method that returns a dict of outputs.
    """

    def __init__(self):
        # Keep init lightweight; heavy work happens lazily in run()
        self._pipe_info = None

    def _ensure_pipeline(self):
        if self._pipe_info is None:
            self._pipe_info = _load_pipeline()
        return self._pipe_info

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
        pipe_info = self._ensure_pipeline()
        pipeline = pipe_info["pipeline"]
        torch = pipe_info["torch"]
        device = pipe_info["device"]

        width = max(64, min(2048, int(width)))
        height = max(64, min(2048, int(height)))
        steps = max(1, min(200, int(steps)))
        guidance_scale = float(guidance_scale)

        generator = None
        if seed is not None and int(seed) >= 0:
            seed = int(seed)
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

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
                import numpy as np
                arr = img.cpu().permute(1, 2, 0).numpy()
                arr = (arr * 255).round().astype("uint8")
                img = Image.fromarray(arr)
            except Exception:
                raise RuntimeError("Unable to convert pipeline output to PIL Image")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": encoded}
