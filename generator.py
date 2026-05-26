# models/hunyuan_t2i_turbo/generator.py
# Minimal Modly-compatible generator for HunyuanDiT-Turbo
# Lazy-loads the diffusers pipeline to avoid locking files during install

import os
import io
import base64
from typing import Optional, Dict, Any

# Lightweight imports only; heavy libs are imported inside _load_pipeline()
from PIL import Image

# Path where setup.py downloads the model repo
MODEL_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "hunyuan_t2i_turbo")
MODEL_LOCAL_DIR = os.path.normpath(MODEL_LOCAL_DIR)

# Cached pipeline reference
_pipeline = None

def _load_pipeline():
    """
    Lazy-load the diffusers pipeline from the local model directory.
    This avoids importing heavy libraries at module import time.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    # Import heavy dependencies here
    import torch
    from diffusers import DiffusionPipeline

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Try to load from local model dir first
    model_dir = MODEL_LOCAL_DIR
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"HunyuanDiT-Turbo model not found at {model_dir}. Run setup install first.")

    # Load pipeline
    # Use low_cpu_mem_usage and torch_dtype where supported
    pipeline = DiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None
    )

    # Move to device
    pipeline = pipeline.to(device)

    _pipeline = {
        "pipeline": pipeline,
        "device": device,
        "torch": torch
    }
    return _pipeline

class Generator:
    """
    Modly generator wrapper.
    Exposes run(prompt, negative_prompt, steps, guidance_scale, width, height, seed)
    Returns dict with key "image" containing base64 PNG.
    """

    def __init__(self):
        # Nothing heavy here; pipeline is lazy-loaded in run()
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
        """
        Generate an image from prompt and return base64 PNG in {"image": "<base64>"}.
        """

        pipe_info = self._ensure_pipeline()
        pipeline = pipe_info["pipeline"]
        torch = pipe_info["torch"]
        device = pipe_info["device"]

        # Validate and clamp sizes to reasonable values
        width = max(64, min(2048, int(width)))
        height = max(64, min(2048, int(height)))
        steps = max(1, min(200, int(steps)))
        guidance_scale = float(guidance_scale)

        # Prepare generator for reproducibility
        generator = None
        if seed is not None and int(seed) >= 0:
            seed = int(seed)
            if device == "cuda":
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed)

        # Run the pipeline
        # Use kwargs that are commonly supported; if the local pipeline has different API adapt accordingly
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

        # result.images is expected; adapt if pipeline returns different structure
        images = getattr(result, "images", None)
        if images is None:
            # Try common alternative keys
            images = result if isinstance(result, list) else []

        if not images:
            raise RuntimeError("Pipeline returned no images")

        # Take first image
        img = images[0]
        if not isinstance(img, Image.Image):
            # If it's a tensor, convert to PIL
            try:
                import numpy as np
                arr = img.cpu().permute(1, 2, 0).numpy()
                arr = (arr * 255).round().astype("uint8")
                img = Image.fromarray(arr)
            except Exception:
                raise RuntimeError("Unable to convert pipeline output to PIL Image")

        # Encode to PNG base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": encoded}
