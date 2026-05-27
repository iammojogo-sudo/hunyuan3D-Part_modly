"""
generator.py

Modly-compatible generator for HunyuanImage-2.1 text-to-image.

Key behaviors:
- Uses Modly local models dir: ~/.modly/models/<model_id> by default.
- Implements is_downloaded() and download_weights() using huggingface_hub.snapshot_download.
- Loads diffusers pipeline from local model folder with local_files_only=True.
- Exposes HunyuanImage21Generator with load(), unload(), generate(params, progress_cb, cancel_event).
"""

import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple

import torch
from PIL import Image

# Try to import pipeline class; if unavailable, raise clear error at load time.
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

    # -------------------------
    # Model path helpers
    # -------------------------
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

    def _downloaded_root_from_snapshot(self, snapshot_root: Path) -> Path:
        """
        snapshot_download may place files under snapshot_root/<repo_id> or directly under snapshot_root.
        Normalize to the directory that contains the expected subfolders (base/, vae/, etc).
        """
        # If snapshot_root contains subfolder with repo name, prefer that
        repo_name = self.HF_REPO.split("/")[-1]
        candidate = snapshot_root / repo_name
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
        # Otherwise, if snapshot_root itself looks like the repo root, return it
        if any((snapshot_root / p).exists() for p in ("base", "vae", "text_encoder")):
            return snapshot_root
        # Otherwise return snapshot_root and let load fail with clear error
        return snapshot_root

    # -------------------------
    # Download helpers
    # -------------------------
    def is_downloaded(self) -> bool:
        """
        Return True if the model files exist in the Modly models folder for this node.
        Uses the same file checks as the manifest's download_check.
        """
        target = self._target_model_path()
        checks = ["base/config.json", "base/pytorch_model.safetensors", "vae/config.json"]
        for c in checks:
            if not (target / c).exists():
                return False
        return True

    def download_weights(self, progress_cb: ProgressCallback = None) -> Tuple[bool, str]:
        """
        Download HF repo into Modly models folder for this node using snapshot_download.
        Returns (success, message).
        """
        if not _HAS_HF_HUB:
            return False, "huggingface_hub not installed in extension environment"

        target = self._target_model_path()
        target.mkdir(parents=True, exist_ok=True)

        # snapshot_download will populate cache_dir; we pass target as cache_dir.
        # snapshot_download may create a subfolder named after the repo; normalize after download.
        try:
            if progress_cb:
                progress_cb(0.0, "starting download from Hugging Face")
            # Use repo_id and cache_dir=target. repo_type default is "model".
            snapshot_dir = snapshot_download(repo_id=self.HF_REPO, cache_dir=str(target), repo_type="model")
            # snapshot_dir is the path where HF placed the files (string)
            snapshot_path = Path(snapshot_dir)
            root = self._downloaded_root_from_snapshot(snapshot_path)
            # If snapshot placed files under target/<repo_name>, move them up to target root for consistent layout
            if root != target:
                # move contents of root into target
                for item in root.iterdir():
                    dest = target / item.name
                    if dest.exists():
                        # remove existing to avoid partial conflicts
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    if item.is_dir():
                        shutil.move(str(item), str(dest))
                    else:
                        shutil.move(str(item), str(dest))
                # if root is a subfolder under target, remove it if empty
                try:
                    if root.exists() and root != target:
                        shutil.rmtree(root)
                except Exception:
                    pass

            if progress_cb:
                progress_cb(1.0, "download complete")
            # verify
            if self.is_downloaded():
                return True, f"Downloaded to {str(target)}"
            else:
                return False, f"Downloaded but required files not found under {str(target)}"
        except Exception as e:
            return False, f"Download failed: {e}"

    # -------------------------
    # Load / Unload / Generate
    # -------------------------
    def load(self, progress_cb: ProgressCallback = None) -> None:
        """
        Load the diffusers pipeline. Prefer local model folder; if missing, attempt to download automatically.
        """
        if self.pipe is not None:
            return

        target = self._target_model_path()

        # If not downloaded, attempt to download (Modly's UI also triggers download; this is a fallback)
        if not self.is_downloaded():
            ok, msg = self.download_weights(progress_cb=progress_cb)
            if not ok:
                # If download failed, try remote load as last resort (requires internet)
                if progress_cb:
                    progress_cb(0.0, f"local download failed: {msg}; attempting remote load")
            else:
                if progress_cb:
                    progress_cb(0.0, f"model ready at {str(target)}")

        # Try to load from local target first (local_files_only=True)
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
                # fallback to HF remote
                try:
                    self.pipe = HunyuanImagePipeline.from_pretrained(
                        self.HF_REPO,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    ).to(self.device)
                    self._loaded_from = f"hf:{self.HF_REPO}"
                except Exception as e_remote:
                    raise RuntimeError(f"Failed to load pipeline locally ({e_local}) and remotely ({e_remote})")
        else:
            # No local files; try remote HF load
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

        # optional perf tweaks
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
        """
        Generate an image and return {"image_path": ..., "meta": {...}}
        """
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
