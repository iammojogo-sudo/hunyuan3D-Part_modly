import io
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress

# Redirect print to stderr so stdout stays clean for the JSON runner protocol.
_print = print


def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)


_HF_REPO_ID = "Tencent-Hunyuan/HunyuanImage-2.1"


def _safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class HunyuanImage21Generator(BaseGenerator):
    MODEL_ID     = "hunyuan_image_2_1_t2i"
    DISPLAY_NAME = "HunyuanImage 2.1 Text-to-Image"
    VRAM_GB      = 6

    # ------------------------------------------------------------------
    # Download checks
    # ------------------------------------------------------------------

    def is_downloaded(self):
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return (self.model_dir / "base" / "config.json").exists()

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        import torch

        try:
            from diffusers import HunyuanImagePipeline
        except ImportError:
            raise RuntimeError(
                "diffusers HunyuanImagePipeline not available; "
                "run Repair on the extension to reinstall dependencies."
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype  = torch.float16 if self._device == "cuda" else torch.float32

        print("[HunyuanImage21Generator] Loading pipeline from %s ..." % self.model_dir)

        try:
            pipe = HunyuanImagePipeline.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                torch_dtype=self._dtype,
            ).to(self._device)
        except Exception as e_local:
            print("[HunyuanImage21Generator] Local load failed (%s), trying HF remote ..." % e_local)
            pipe = HunyuanImagePipeline.from_pretrained(
                _HF_REPO_ID,
                torch_dtype=self._dtype,
            ).to(self._device)

        try:
            if self._device == "cuda":
                pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._model = pipe
        print("[HunyuanImage21Generator] Loaded on %s." % self._device)

    def unload(self):
        self._model  = None
        self._device = None
        self._dtype  = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}

        prompt = params.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required")

        negative_prompt = params.get("negative_prompt") or None
        width           = _safe_int(params.get("width"), 1024)
        height          = _safe_int(params.get("height"), 1024)
        steps           = _safe_int(params.get("steps"), 30)
        guidance_scale  = _safe_float(params.get("guidance_scale"), 5.0)
        seed_val        = _safe_int(params.get("seed"), 0)
        if seed_val == 0:
            seed_val = random.randint(1, 2 ** 31 - 1)

        self._report(progress_cb, 5, "Starting generation ...")
        self._check_cancelled(cancel_event)

        generator = torch.Generator(device=self._device).manual_seed(seed_val)

        self._report(progress_cb, 10, "Generating image ...")

        stop_evt      = threading.Event()
        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 95, "Generating image ...", stop_evt),
                daemon=True,
            )
            progress_thread.start()

        try:
            with torch.inference_mode():
                out = self._model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )
            image = out.images[0]
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 98, "Saving image ...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_name = "hunyuan21_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
        out_path = self.outputs_dir / out_name
        image.save(str(out_path), format="PNG")

        self._report(progress_cb, 100, "Done")
        print("[HunyuanImage21Generator] Saved to %s" % out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        repo_id        = self.hf_repo or _HF_REPO_ID
        manifest_skips = list(getattr(self, "hf_skip_prefixes", []) or [])
        ignore = []
        for pattern in manifest_skips:
            ignore.append(pattern)
            if isinstance(pattern, str) and pattern.endswith("/"):
                ignore.append(pattern + "*")
        ignore += ["*.md", "*.txt", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"]

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[HunyuanImage21Generator] Downloading weights from %s ..." % repo_id)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.model_dir),
            ignore_patterns=ignore,
        )
        print("[HunyuanImage21Generator] Weights downloaded to %s." % self.model_dir)
