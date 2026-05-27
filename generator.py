"""
HunyuanDiT Turbo - Text-to-Image generator for Modly.

Single node: generate (text -> image)
The prompt comes from the connected Text input node via image_bytes.
Generation settings (steps, size, seed) are in params_schema.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Optional
import threading

# ------------------------------------------------------------------ #
#  Inject extension venv site-packages (Windows + Linux/macOS)       #
# ------------------------------------------------------------------ #
_EXT_DIR = Path(__file__).parent
for _venv in (_EXT_DIR / "venv", _EXT_DIR / ".venv"):
    if _venv.exists():
        _win_sp = _venv / "Lib" / "site-packages"
        if _win_sp.exists() and str(_win_sp) not in sys.path:
            sys.path.insert(0, str(_win_sp))
        _lib = _venv / "lib"
        if _lib.exists():
            for _pydir in _lib.glob("python3.*"):
                _sp = _pydir / "site-packages"
                if _sp.exists() and str(_sp) not in sys.path:
                    sys.path.insert(0, str(_sp))
        break

from services.generators.base import BaseGenerator


def _get_hf_token():
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _extract_prompt(image_bytes):
    """
    Modly passes the connected text node's content as image_bytes.
    Try to decode it as UTF-8 text. If it looks like a file path,
    read that file instead. Fall back to empty string on any failure.
    """
    if not image_bytes:
        return ""
    try:
        text = image_bytes.decode("utf-8").strip()
        # If it decoded cleanly and looks like a file path, read it
        p = Path(text)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8").strip()
        return text
    except Exception:
        return ""


class HunyuanT2IGenerator(BaseGenerator):

    MODEL_ID     = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB      = 6

    # ---------------------------------------------------------------- #
    #  Download / lifecycle                                             #
    # ---------------------------------------------------------------- #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "model_index.json").exists()

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print("[HunyuanT2I] Downloading weights to {} ...".format(self.model_dir))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
            local_dir=str(self.model_dir),
            local_dir_use_symlinks=False,
            token=_get_hf_token(),
        )
        print("[HunyuanT2I] Download complete.")

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        import torch
        from diffusers import HunyuanDiTPipeline
        print("[HunyuanT2I] Loading pipeline from {} ...".format(self.model_dir))
        pipe = HunyuanDiTPipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        self._model = pipe
        print("[HunyuanT2I] Pipeline ready.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---------------------------------------------------------------- #
    #  Inference                                                        #
    # ---------------------------------------------------------------- #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        params = params or {}

        # Prompt comes from the connected text node via image_bytes
        prompt = _extract_prompt(image_bytes)
        if not prompt:
            prompt = "a beautiful landscape"

        steps  = int(params.get("num_inference_steps", 20))
        seed   = int(params.get("seed", -1))
        height = int(params.get("height", 1024))
        width  = int(params.get("width", 1024))

        print("[HunyuanT2I] prompt='{}' steps={} seed={} {}x{}".format(
            prompt, steps, seed, width, height
        ))

        self._report(progress_cb, 5, "Loading model ...")
        if self._model is None:
            self.load()

        self._report(progress_cb, 20, "Generating image ...")

        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        result = self._model(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=generator,
        )

        self._report(progress_cb, 95, "Saving ...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = "{}_{}.png".format(int(time.time()), uuid.uuid4().hex[:8])
        out  = self.outputs_dir / name
        result.images[0].save(str(out), format="PNG")

        self._report(progress_cb, 100, "Done")
        print("[HunyuanT2I] saved -> {}".format(out))
        return out
