"""
generator.py

Modly-compatible generator for HunyuanImage-2.1 text-to-image.
Exposes HunyuanImage21Generator with load(), unload(), generate(params, progress_cb, cancel_event).

Behavior:
- Prefer loading from a local model directory (Modly downloads HF repos locally).
- If local files are missing, fall back to from_pretrained HF repo.
- Try to use a diffusers pipeline class if available; otherwise attempt manual assembly.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import torch
from PIL import Image

# Try to import a HunyuanImage pipeline from diffusers; if not present, we'll fall back.
try:
    from diffusers import HunyuanImagePipeline  # type: ignore
    _HAS_PIPELINE = True
except Exception:
    HunyuanImagePipeline = None  # type: ignore
    _HAS_PIPELINE = False

# Helpful types
ProgressCallback = Optional[Callable[[float, str], None]]


class HunyuanImage21Generator:
    """
    Generator that matches Modly node schema:
      - load()
      - unload()
      - generate(params, progress_cb, cancel_event) -> {"image_path":..., "meta": {...}}
    """

    HF_REPO = "Tencent-Hunyuan/HunyuanImage-2.1"
    DEFAULT_OUTPUT_DIR = Path("./outputs")

    def __init__(self, model_dir: Optional[str] = None, outputs_dir: Optional[str] = None):
        """
        model_dir: optional local path where Modly downloaded the HF repo (preferred).
        If not provided, the generator will attempt to load from HF repo directly.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(model_dir) if model_dir else None
        self.outputs_dir = Path(outputs_dir or self.DEFAULT_OUTPUT_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self._loaded_from = None

    def _local_subpath(self, *parts) -> Path:
        if not self.model_dir:
            raise FileNotFoundError("model_dir not set")
        return self.model_dir.joinpath(*parts)

    def _has_local_base(self) -> bool:
        if not self.model_dir:
            return False
        # Check for a minimal file that should exist in the base folder
        base = self._local_subpath("base")
        return base.exists() and any(base.glob("**/*"))

    def _load_pipeline_from_local(self):
        """
        Attempt to load a pipeline or components from the local model_dir.
        This prefers a diffusers pipeline if available; otherwise uses from_pretrained with local folder.
        """
        if not self.model_dir:
            raise FileNotFoundError("No local model_dir provided")

        # If diffusers provides HunyuanImagePipeline, try to load it from the local folder
        if _HAS_PIPELINE and HunyuanImagePipeline is not None:
            try:
                # If Modly placed the repo root in model_dir, pass that path
                self.pipe = HunyuanImagePipeline.from_pretrained(
                    str(self.model_dir),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                self._loaded_from = f"local_pipeline:{self.model_dir}"
                return
            except Exception:
                # Fall through to trying subfolders or remote
                pass

        # Fallback: try to load from the 'base' subfolder specifically
        base_path = self._local_subpath("base")
        if base_path.exists():
            try:
                if _HAS_PIPELINE and HunyuanImagePipeline is not None:
                    self.pipe = HunyuanImagePipeline.from_pretrained(
                        str(base_path),
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    ).to(self.device)
                    self._loaded_from = f"local_base:{base_path}"
                    return
            except Exception:
                pass

        # If we reach here, local loading failed
        raise RuntimeError("Local model files found but failed to load pipeline from them")

    def _load_pipeline_from_hf(self):
        """
        Load pipeline directly from HF repo (internet required).
        """
        if not _HAS_PIPELINE or HunyuanImagePipeline is None:
            raise RuntimeError("HunyuanImagePipeline not available in diffusers; install a compatible diffusers version")
        self.pipe = HunyuanImagePipeline.from_pretrained(
            self.HF_REPO,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._loaded_from = f"hf:{self.HF_REPO}"

    def load(self) -> None:
        """
        Load model pipeline. Prefer local model_dir (if provided or via env var MODEL_DIR).
        """
        if self.pipe is not None:
            return

        # Allow Modly to pass model_dir via env var if it sets one
        env_model_dir = os.environ.get("MODEL_DIR") or os.environ.get("MODLY_MODEL_DIR")
        if env_model_dir and not self.model_dir:
            self.model_dir = Path(env_model_dir)

        # Try local first
        try:
            if self.model_dir and self._has_local_base():
                self._load_pipeline_from_local()
            else:
                # No local base; attempt HF remote load
                self._load_pipeline_from_hf()
        except Exception as e_local:
            # If local load failed, try HF remote as fallback
            try:
                self._load_pipeline_from_hf()
            except Exception as e_hf:
                # Raise a combined error for easier debugging
                raise RuntimeError(f"Failed to load model locally ({e_local}) and from HF ({e_hf})")

        # Optional performance tweaks
        try:
            if self.device == "cuda":
                # enable xformers if available
                self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # Disable internal progress bars (Modly provides its own)
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    def unload(self) -> None:
        self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate(
        self,
        params: Dict[str, Any],
        progress_cb: ProgressCallback = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate an image.

        params keys (as in manifest node schema):
          - prompt (required)
          - negative_prompt
          - width
          - height
          - steps
          - guidance_scale
          - seed

        Returns:
          {"image_path": "<path>", "meta": {...}}
        """
        self.load()
        if self.pipe is None:
            raise RuntimeError("pipeline not loaded")

        prompt = params.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required")

        negative_prompt = params.get("negative_prompt", None)
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        steps = int(params.get("steps", 30))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        seed = int(params.get("seed", 0)) or random.randint(1, 2**31 - 1)

        if progress_cb:
            progress_cb(0.0, "starting")

        if cancel_event is not None and getattr(cancel_event, "is_set", None):
            if cancel_event.is_set():
                raise RuntimeError("generation cancelled before start")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run inference
        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        image: Image.Image = out.images[0]

        out_name = f"hunyuan_{seed}.png"
        out_path = self.outputs_dir / out_name
        image.save(out_path, format="PNG")

        if progress_cb:
            progress_cb(1.0, "done")

        meta = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "loaded_from": self._loaded_from,
        }

        return {"image_path": str(out_path), "meta": meta}


# Module-level helper used by some Modly patterns
def generate(params: Dict[str, Any], progress_cb: ProgressCallback = None, cancel_event: Optional[Any] = None) -> Dict[str, Any]:
    gen = HunyuanImage21Generator()
    return gen.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)
