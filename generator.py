"""
HunyuanDiT Turbo - Text-to-Image generator for Modly.

The prompt lives in params_schema so Modly renders it as a text field in the
sidebar. image_bytes is unused (T2I has no input image).
"""
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Optional

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


class HunyuanT2IGenerator(BaseGenerator):

    MODEL_ID    = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB     = 6

    # ---------------------------------------------------------------- #
    #  Lifecycle                                                        #
    # ---------------------------------------------------------------- #
    def is_downloaded(self) -> bool:
        return (self.model_dir / "model_index.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        import torch
        from diffusers import HunyuanDiTPipeline
        print("[HunyuanT2I] Loading pipeline from {} ...".format(self.model_dir))
        pipe = HunyuanDiTPipeline.from_pretrained(
            str(self.model_dir), torch_dtype=torch.float16
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
    ) -> Path:
        import torch
        params = params or {}

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            prompt = "a beautiful landscape"

        steps  = int(params.get("num_inference_steps", 20))
        seed   = int(params.get("seed", -1))
        height = int(params.get("height", 1024))
        width  = int(params.get("width", 1024))

        print("[HunyuanT2I] prompt='{}' steps={} seed={}".format(prompt, steps, seed))

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

    # ---------------------------------------------------------------- #
    #  Download helper (called from load() if weights are missing)     #
    # ---------------------------------------------------------------- #
    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print("[HunyuanT2I] Downloading weights to {} ...".format(self.model_dir))
        snapshot_download(
            repo_id="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
            local_dir=str(self.model_dir),
            local_dir_use_symlinks=False,
            token=_get_hf_token(),
        )
        print("[HunyuanT2I] Download complete.")

    # ---------------------------------------------------------------- #
    #  UI params — rendered by Modly in the sidebar                    #
    # ---------------------------------------------------------------- #
    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id": "prompt",
                "label": "Prompt",
                "type": "string",
                "default": "a beautiful landscape",
            },
            {
                "id": "num_inference_steps",
                "label": "Quality",
                "type": "select",
                "default": 20,
                "options": [
                    {"value": 10, "label": "Fast (10 steps)"},
                    {"value": 20, "label": "Balanced (20 steps)"},
                    {"value": 30, "label": "High (30 steps)"},
                ],
            },
            {
                "id": "height",
                "label": "Height",
                "type": "select",
                "default": 1024,
                "options": [
                    {"value": 512,  "label": "512 px"},
                    {"value": 768,  "label": "768 px"},
                    {"value": 1024, "label": "1024 px"},
                ],
            },
            {
                "id": "width",
                "label": "Width",
                "type": "select",
                "default": 1024,
                "options": [
                    {"value": 512,  "label": "512 px"},
                    {"value": 768,  "label": "768 px"},
                    {"value": 1024, "label": "1024 px"},
                ],
            },
            {
                "id": "seed",
                "label": "Seed  (-1 = random)",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 2147483647,
            },
        ]
