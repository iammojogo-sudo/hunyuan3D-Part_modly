"""
generator.py

Modly-compatible generator that:
- exposes module-level download hooks Modly calls,
- downloads HF repo into ~/.modly/models/<model_id>/ using snapshot_download,
- normalizes snapshot layout so base/, vae/, etc. are directly under the model folder,
- loads diffusers pipeline with local_files_only=True and falls back to HF remote if needed,
- exposes load(), unload(), generate() on the generator class and module-level helpers.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple

import torch
from PIL import Image

# Try to import pipeline class from diffusers
try:
    from diffusers import HunyuanImagePipeline  # type: ignore
    _HAS_PIPELINE = True
except Exception:
    HunyuanImagePipeline = None  # type: ignore
    _HAS_PIPELINE = False

# huggingface_hub for snapshot_download
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

    # Path helpers
    def _resolve_models_dir(self) -> Path:
        if self.model_dir:
            return self.model_dir
        env = os.environ.get("MODLY_MODELS_DIR") or os.environ.get("MODEL_DIR") or os.environ.get("MODLY_MODEL_DIR")
        if env:
            return Path(env)
        return self.DEFAULT_MODELS_DIR

    def _target_model_path(self, model_id: Optional[str] = None) -> Path:
        base = self._resolve_models_dir()
        mid = model_id if model_id else self.NODE_MODEL_ID
        return base / mid

    def _downloaded_root_from_snapshot(self, snapshot_root: Path) -> Path:
        repo_name = self.HF_REPO.split("/")[-1]
        candidate = snapshot_root / repo_name
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
        if any((snapshot_root / p).exists() for p in ("base", "vae", "text_encoder")):
            return snapshot_root
        return snapshot_root

    # Download checks
    def is_downloaded(self, model_id: Optional[str] = None) -> bool:
        target = self._target_model_path(model_id)
        checks = ["base/config.json", "base/pytorch_model.safetensors", "vae/config.json"]
        for c in checks:
            if not (target / c).exists():
                return False
        return True

    def download_weights(self, model_id: Optional[str] = None, progress_cb: ProgressCallback = None) -> Tuple[bool, str]:
        if not _HAS_HF_HUB:
            return False, "huggingface_hub not installed in extension venv"

        target = self._target_model_path(model_id)
        target.mkdir(parents=True, exist_ok=True)

        try:
            if progress_cb:
                progress_cb(0.0, "starting download from Hugging Face")
            snapshot_dir = snapshot_download(repo_id=self.HF_REPO, cache_dir=str(target), repo_type="model")
            snapshot_path = Path(snapshot_dir)
            root = self._downloaded_root_from_snapshot(snapshot_path)

            if root != target:
                for item in root.iterdir():
                    dest = target / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                try:
                    if root.exists() and root != target:
                        shutil.rmtree(root)
                except Exception:
                    pass

            if progress_cb:
                progress_cb(1.0, "download complete")

            if self.is_downloaded(model_id):
                return True, f"Downloaded to {str(target)}"
            else:
                return False, f"Downloaded but required files not found under {str(target)}"
        except Exception as e:
            return False, f"Download failed: {e}"

    # Load / Unload / Generate
    def load(self, model_id: Optional[str] = None, progress_cb: ProgressCallback = None) -> None:
        if self.pipe is not None:
            return

        target = self._target_model_path(model_id)

        if not self.is_downloaded(model_id):
            ok, msg = self.download_weights(model_id=model_id, progress_cb=progress_cb)
            if not ok:
                if progress_cb:
                    progress_cb(0.0, f"local download failed: {msg}; attempting remote load")
            else:
                if progress_cb:
                    progress_cb(0.0, f"model ready at {str(target)}")

        if target.exists() and any(target.iterdir()):
            if not _HAS_PIPELINE:
                raise RuntimeError("diffusers HunyuanImagePipeline not available; install compatible diffusers")
            try:
                self.pipe = HunyuanImagePipeline.from_pretrained(
                    str(target),
                    local_files_only=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                self._loaded_from = f"local:{str(target)}"
            except Exception as e_local:
                try:
                    self.pipe = HunyuanImagePipeline.from_pretrained(
                        self.HF_REPO,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    ).to(self.device)
                    self._loaded_from = f"hf:{self.HF_REPO}"
                except Exception as e_remote:
                    raise RuntimeError(f"Failed to load pipeline locally ({e_local}) and remotely ({e_remote})")
        else:
            if not _HAS_PIPELINE:
                raise RuntimeError("diffusers HunyuanImagePipeline not available; install compatible diffusers")
            try:
                self.pipe = HunyuanImagePipeline.from_pretrained(
                    self.HF_REPO,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                self._loaded_from = f"hf:{self.HF_REPO}"
            except Exception as e:
                raise RuntimeError(f"Failed to load pipeline from HF repo {self.HF_REPO}: {e}")

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
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.load(model_id=model_id, progress_cb=progress_cb)
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
            "model_id": model_id or self.NODE_MODEL_ID,
        }

        return {"image_path": str(out_path), "meta": meta}


# Module-level helpers for Modly backend
_default_gen = HunyuanImage21Generator()


def is_downloaded_for_model(model_id: Optional[str] = None) -> bool:
    return _default_gen.is_downloaded(model_id=model_id)


def download_weights_for_model(model_id: Optional[str] = None, progress_cb: ProgressCallback = None) -> Tuple[bool, str]:
    return _default_gen.download_weights(model_id=model_id, progress_cb=progress_cb)


def generate(params: Dict[str, Any], progress_cb: ProgressCallback = None, cancel_event: Optional[Any] = None) -> Dict[str, Any]:
    return _default_gen.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)
