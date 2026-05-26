"""
Hunyuan T2I Turbo generator for Modly.

This generator:
 - Ensures model files are present (downloads via huggingface_hub.snapshot_download)
   and supports authenticated downloads using an HF token from environment.
 - Loads a Diffusers-style pipeline from the local model directory.
 - Exposes generate(prompt_bytes, params, progress_cb, cancel_event) returning a PNG path.

Notes:
 - Provide a Hugging Face token via HUGGINGFACE_HUB_TOKEN or HF_TOKEN if the repo is gated.
 - The generator expects a BaseGenerator implementation in services.generators.base.
"""
import base64
import io
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress


_HF_REPO = "TencentARC/HunyuanDiT-Turbo"
_DOWNLOAD_ATTEMPTS = 3
_GLB_MAGIC = b"glTF"


def _get_hf_token() -> Optional[str]:
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


class HunyuanT2IGenerator(BaseGenerator):
    MODEL_ID = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB = 6
    MODEL_VARIANT = "t2i-turbo"

    def is_downloaded(self) -> bool:
        # Check for a minimal marker file in model_dir
        marker = self.model_dir / "model_index.json"
        return marker.exists()

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        from httpx import HTTPStatusError

        repo_id = self.hf_repo or _HF_REPO
        token = _get_hf_token()
        last_exc = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(self.model_dir),
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
        raise RuntimeError("Failed to download model after %d attempts: %s" % (_DOWNLOAD_ATTEMPTS, last_exc))

    def _ensure_model_present(self):
        if not self.is_downloaded():
            self._download_weights()

    def load(self):
        if getattr(self, "_model_loaded", False):
            return

        self._ensure_model_present()

        # Import and load pipeline
        try:
            from diffusers import DiffusionPipeline
            import torch
        except Exception as exc:
            raise RuntimeError("Required packages not installed in extension venv: %s" % exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Load pipeline from local model_dir
        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                torch_dtype=dtype,
            )
            self._pipe.to(device)
        except Exception as exc:
            raise RuntimeError("Failed to load pipeline from %s: %s" % (self.model_dir, exc)) from exc

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
                # Many diffusers pipelines accept height/width via kwargs; pass if supported.
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

        # Save output
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8]))
        image.save(str(out_path))
        self._report(progress_cb, 100, "Done")
        return str(out_path)
