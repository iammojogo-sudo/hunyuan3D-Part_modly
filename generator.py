import os
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Optional
import threading

# pull in venv packages
_here = Path(__file__).parent
for _v in (_here / "venv", _here / ".venv"):
    if _v.exists():
        _sp = _v / "Lib" / "site-packages"
        if _sp.exists() and str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))
        _lib = _v / "lib"
        if _lib.exists():
            for _d in _lib.glob("python3.*"):
                _s = _d / "site-packages"
                if _s.exists() and str(_s) not in sys.path:
                    sys.path.insert(0, str(_s))
        break

from services.generators.base import BaseGenerator


def _hf_token():
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


class HunyuanT2IGenerator(BaseGenerator):

    MODEL_ID     = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB      = 6

    def is_downloaded(self):
        return (self.model_dir / "model_index.json").exists()

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[t2i] downloading weights to", self.model_dir)
        snapshot_download(
            repo_id="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
            local_dir=str(self.model_dir),
            local_dir_use_symlinks=False,
            token=_hf_token(),
        )

    def load(self):
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        import torch
        from diffusers import HunyuanDiTPipeline
        print("[t2i] loading pipeline...")
        pipe = HunyuanDiTPipeline.from_pretrained(
            str(self.model_dir), torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        self._model = pipe
        print("[t2i] ready")

    def unload(self):
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        params = params or {}

        # image_bytes comes from whatever image node is connected but we don't use it
        # the actual prompt lives in params
        prompt = str(params.get("prompt", "")).strip() or "a beautiful landscape"
        steps  = int(params.get("num_inference_steps", 20))
        seed   = int(params.get("seed", -1))
        height = int(params.get("height", 1024))
        width  = int(params.get("width", 1024))

        print("[t2i] prompt='{}' steps={} seed={}".format(prompt, steps, seed))

        self._report(progress_cb, 5, "Loading model...")
        if self._model is None:
            self.load()

        self._report(progress_cb, 20, "Generating...")

        gen = None
        if seed >= 0:
            gen = torch.Generator(device="cuda").manual_seed(seed)

        out_images = self._model(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=gen,
        )

        self._report(progress_cb, 95, "Saving...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        fname = "{}_{}.png".format(int(time.time()), uuid.uuid4().hex[:8])
        out = self.outputs_dir / fname
        out_images.images[0].save(str(out), format="PNG")

        self._report(progress_cb, 100, "Done")
        print("[t2i] saved ->", out)
        return out
