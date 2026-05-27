"""
generator.py

Modly-compatible generator for HunyuanImage-2.1 text-to-image.

Behavior:
- Prefer local model folder under Modly models dir: ~/.modly/models/<model_id> (or MODLY_MODELS_DIR env).
- If local files are missing, automatically download the HF repo into that model folder using huggingface_hub.snapshot_download.
- Load a diffusers pipeline (HunyuanImagePipeline) from the local folder if present, otherwise from HF.
- Exposes HunyuanImage21Generator with load(), unload(), generate(params, progress_cb, cancel_event).
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import torch
from PIL import Image

# Attempt to import diffusers pipeline class; if unavailable, the code will raise a clear error.
try:
    from diffusers import HunyuanImagePipeline  # type: ignore
    _HAS_PIPELINE = True
except Exception:
    HunyuanImagePipeline = None  # type: ignore
    _HAS_PIPELINE = False

# huggingface_hub for programmatic download
try:
    from huggingface_hub import snapshot_download  # type: ignore
    _HAS_HF_HUB = True
except Exception:
    snapshot_download = None  # type: ignore
    _HAS_HF_HUB = False

ProgressCallback = Optional[Callable[[float, str], None]]


class HunyuanImage21Generator:
    HF_REPO = "Tencent-Hunyuan/HunyuanImage-2.1"
    NODE_MODEL_ID = "hunyuan_image_2_1_t2i/generate"
    DEFAULT_MODELS_DIR = Path.home() / ".modly" / "models"
    DEFAULT_OUTPUTS_DIR = Path("./outputs")

    def __init__(self, model_dir: Optional[str] = None, outputs_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(model_dir) if model_dir else None
        self.outputs_dir = Path(outputs_dir or self.DEFAULT_OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self._loaded_from = None

    def _resolve_models_dir(self) -> Path:
        # Priority: explicit model_dir -> MODLY_MODELS_DIR env -> default ~/.modly/models
        if self.model_dir:
            return self.model_dir
        env = os.environ.get("MODLY_MODELS_DIR") or os.environ.get("MODEL_DIR") or os.environ.get("MODLY_MODEL_DIR")
        if env:
            return Path(env)
        return self.DEFAULT_MODELS_DIR

    def _target_model_path(self) -> Path:
        base = self._resolve_models_dir()
        return base / self.NODE_MODEL_ID

    def _check_downloaded(self, target: Path) -> bool:
        # Files to check that indicate a successful download
        checks = ["base/config.json", "base/pytorch_model.safetensors", "vae/config.json"]
        for c in checks:
            if not (target / c).exists():
                return False
        return True

    def _download_to_local(self, progress_cb: ProgressCallback = None) -> None:
        if not _HAS_HF_HUB:
            raise RuntimeError("huggingface_hub is not installed in the extension environment; run setup to install it.")
        target = self._target_model_path()
        target.mkdir(parents=True, exist_ok=True)
        # snapshot_download will populate the cache_dir; we use repo_id and let it place files under cache_dir/repo_id
        # To ensure files end up under target, we call snapshot_download with repo_id and allow_patterns=None and set cache_dir to target.
        try:
            if progress_cb:
                progress_cb(0.0, "downloading model from Hugging Face")
            snapshot_download(repo_id=self.HF_REPO, cache_dir=str(target), repo_type="model")
            if progress_cb:
                progress_cb(1.0, "download complete")
        except Exception as e:
            raise RuntimeError(f"Failed to download HF repo {self.HF_REPO}: {e}")

    def load(self, progress_cb: ProgressCallback = None) -> None:
        if self.pipe is not None:
            return

        target = self._target_model_path()

        # If target missing or incomplete, attempt to download automatically
        if not self._check_downloaded(target):
            try:
                self._download_to_local(progress_cb=progress_cb)
            except Exception as e:
                # If automatic download fails, fall back to remote load attempt (requires internet)
                if progress_cb:
                    progress_cb(0.0, f"local download failed: {e}; attempting remote load")
                # remote load will be attempted below

        # Try to load from local target first
        if target.exists() and any(target.iterdir()):
            try:
                if _HAS_PIPELINE and HunyuanImagePipeline is not None:
                    self.pipe = HunyuanImagePipeline.from_pretrained(
                        str(target),
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    ).to(self.device)
                    self._loaded_from = f"local:{target}"
                else:
                    raise RuntimeError("HunyuanImagePipeline not available in diffusers; install a compatible diffusers version.")
            except Exception as e_local:
                # If local load fails, try remote HF load
                try:
                    if _HAS_PIPELINE and HunyuanImagePipeline is not None:
                        self.pipe = HunyuanImagePipeline.from_pretrained(
                            self.HF_REPO,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        ).to(self.device)
                        self._loaded_from = f"hf:{self.HF_REPO}"
                    else:
                        raise RuntimeError("HunyuanImagePipeline not available in diffusers; install a compatible diffusers version.")
                except Exception as e_remote:
                    raise RuntimeError(f"Failed to load pipeline locally ({e_local}) and remotely ({e_remote})")
        else:
            # No local files; try remote HF load
            try:
                if _HAS_PIPELINE and HunyuanImagePipeline is not None:
                    self.pipe = HunyuanImagePipeline.from_pretrained(
                        self.HF_REPO,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    ).to(self.device)
                    self._loaded_from = f"hf:{self.HF_REPO}"
                else:
                    raise RuntimeError("HunyuanImagePipeline not available in diffusers; install a compatible diffusers version.")
            except Exception as e:
                raise RuntimeError(f"Failed to load pipeline from HF repo {self.HF_REPO}: {e}")

        # Optional performance tweaks
        try:
            if self.device == "cuda":
                self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

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
        self.load(progress_cb=progress_cb)
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
            progress_cb(0.0, "starting generation")

        if cancel_event is not None and getattr(cancel_event, "is_set", None):
            if cancel_event.is_set():
                raise RuntimeError("generation cancelled before start")

        generator = torch.Generator(device=self.device).manual_seed(seed)

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
            "model_id": self.NODE_MODEL_ID,
        }

        return {"image_path": str(out_path), "meta": meta}


# Module-level helper used by some Modly patterns
def generate(params: Dict[str, Any], progress_cb: ProgressCallback = None, cancel_event: Optional[Any] = None) -> Dict[str, Any]:
    gen = HunyuanImage21Generator()
    return gen.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)
