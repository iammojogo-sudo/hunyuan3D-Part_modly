"""
Hunyuan T2I Turbo generator for Modly (download target: shared models dir).

This generator ensures model files are present in Modly's shared models directory
and will download them there when the user clicks the purple Download button.

Key points:
 - Uses MODELS_DIR env var when available; otherwise falls back to ~/ModlyData/models.
 - Downloads to a subfolder named after the repo (slashes replaced with underscores).
 - Supports HF auth tokens via HUGGINGFACE_HUB_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN.
 - Loads pipeline from the shared models directory (so Modly's UI and runtime share the same files).
"""
import os
import sys
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress

# Default repo id (change if you want a different default)
DEFAULT_HF_REPO = "TencentARC/HunyuanDiT-Turbo"
_DOWNLOAD_ATTEMPTS = 3


def _get_hf_token() -> Optional[str]:
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _resolve_models_dir() -> Path:
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "ModlyData" / "models"


def _repo_target_dir(repo_id: str) -> Path:
    models_dir = _resolve_models_dir()
    safe_name = repo_id.replace("/", "_")
    return models_dir / safe_name


class HunyuanT2IGenerator(BaseGenerator):
    MODEL_ID = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB = 6
    MODEL_VARIANT = "t2i-turbo"

    def is_downloaded(self) -> bool:
        """
        Check whether the model is present in the shared models directory.
        We look for a minimal marker file (model_index.json) inside the repo subfolder.
        """
        repo_dir = _repo_target_dir(self.hf_repo or DEFAULT_HF_REPO)
        marker = repo_dir / "model_index.json"
        return marker.exists()

    def _download_weights(self):
        """
        Download the HF repo into the shared models directory.
        This is the same location the setup.py download action writes to.
        """
        from huggingface_hub import snapshot_download
        from httpx import HTTPStatusError

        repo_id = self.hf_repo or DEFAULT_HF_REPO
        target_dir = _repo_target_dir(repo_id)
        token = _get_hf_token()

        last_exc = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    use_auth_token=token,
                )
                return
            except HTTPStatusError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 401:
                    raise RuntimeError(
                        "Hugging Face returned 401 Unauthorized while downloading model.\n"
                        "This repository requires authentication. Provide a valid Hugging Face token\n"
                        "via the environment variable HUGGINGFACE_HUB_TOKEN or HF_TOKEN and retry."
                    ) from exc
                last_exc = exc
            except Exception as exc:
                last_exc = exc
            time.sleep(2)
        raise RuntimeError(f"Failed to download model after {_DOWNLOAD_ATTEMPTS} attempts: {last_exc}")

    def _ensure_model_present(self):
        if not self.is_downloaded():
            self._download_weights()

    def load(self):
        if getattr(self, "_model_loaded", False):
            return

        self._ensure_model_present()

        try:
            from diffusers import DiffusionPipeline
            import torch
        except Exception as exc:
            raise RuntimeError("Required packages not installed in extension venv: %s" % exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        repo_dir = _repo_target_dir(self.hf_repo or DEFAULT_HF_REPO)

        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                str(repo_dir),
                local_files_only=True,
                torch_dtype=dtype,
            )
            self._pipe.to(device)
        except Exception as exc:
            raise RuntimeError("Failed to load pipeline from %s: %s" % (repo_dir, exc)) from exc

        self._model_loaded = True
        self._device = device
        print("[HunyuanT2IGenerator] Pipeline ready on %s" % device)

    def unload(self):
        try:
            import torch
            if getattr(self, "_pipe", None) is not None:
                try:
                    del self._pipe
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass
        self._model_loaded = False

    def _report(self, progress_cb, pct, message):
        if progress_cb:
            try:
                progress_cb(pct, message)
            except Exception:
                pass

    def _check_cancelled(self, cancel_event):
        if cancel_event and getattr(cancel_event, "is_set", None):
            if cancel_event.is_set():
                raise RuntimeError("Generation cancelled")

    def generate(self, prompt_bytes, params, progress_cb=None, cancel_event=None):
        """
        prompt_bytes: bytes or str containing the text prompt
        params: dict with optional keys:
            - num_inference_steps (int)
            - seed (int)
            - height (int)
            - width (int)
        Returns: path to generated PNG
        """
        import torch

        prompt = ""
        if isinstance(prompt_bytes, (bytes, bytearray)):
            prompt = prompt_bytes.decode("utf-8", errors="ignore")
        else:
            prompt = str(prompt_bytes or "")

        params = params or {}
        steps = int(params.get("num_inference_steps", 20))
        seed = int(params.get("seed", 42))
        height = int(params.get("height", 512))
        width = int(params.get("width", 512))

        self._report(progress_cb, 5, "Preparing model...")
        self.load()
        self._check_cancelled(cancel_event)

        # Start smooth progress thread while pipeline runs
        stop_evt = threading.Event()
        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 90, "Generating image...", stop_evt),
                daemon=True,
            )
            progress_thread.start()

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed) if hasattr(torch, "Generator") else None
            with torch.no_grad():
                result = self._pipe(
                    prompt,
                    num_inference_steps=steps,
                    generator=generator,
                    height=height,
                    width=width,
                )
                image = result.images[0]
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        # Save output to extension outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8]))
        image.save(str(out_path))
        self._report(progress_cb, 100, "Done")
        return str(out_path)
