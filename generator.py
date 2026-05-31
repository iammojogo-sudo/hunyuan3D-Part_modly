import random
import sys
import threading
import time
import uuid
from pathlib import Path

from services.generators.base import BaseGenerator, smooth_progress

# keep stdout clean for the runner protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

HF_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

def _int(val, default):
    try: return int(val)
    except: return default

def _float(val, default):
    try: return float(val)
    except: return default


class HunyuanDiT12Generator(BaseGenerator):
    MODEL_ID     = "hunyuandit_1_2_t2i"
    DISPLAY_NAME = "HunyuanDiT v1.2 Text-to-Image"
    VRAM_GB      = 6

    def is_downloaded(self):
        check = self.download_check
        if check:
            return (self.model_dir / check).exists()
        return (self.model_dir / "model_index.json").exists()

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        import torch
        from diffusers import HunyuanDiTPipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        print("[HunyuanDiT] loading weights from %s" % self.model_dir)
        pipe = HunyuanDiTPipeline.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            torch_dtype=self._dtype,
        ).to(self._device)

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._model = pipe
        print("[HunyuanDiT] ready on %s" % self._device)

    def unload(self):
        self._model = None
        self._device = None
        self._dtype = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}

        if self._model is None:
            self.load()

        prompt = params.get("prompt", "").strip()
        if not prompt:
            raise ValueError("no prompt — fill in the Prompt field on the node")

        neg    = params.get("negative_prompt") or None
        width  = _int(params.get("width"), 1024)
        height = _int(params.get("height"), 1024)
        steps  = _int(params.get("steps"), 25)
        cfg    = _float(params.get("guidance_scale"), 6.0)
        seed   = _int(params.get("seed"), 0)
        if seed == 0:
            seed = random.randint(1, 2**31 - 1)

        self._report(progress_cb, 5, "starting up")
        self._check_cancelled(cancel_event)

        gen = torch.Generator(device=self._device).manual_seed(seed)

        self._report(progress_cb, 10, "generating")
        stop = threading.Event()
        ticker = None
        if progress_cb:
            ticker = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 95, "generating", stop),
                daemon=True,
            )
            ticker.start()

        try:
            with torch.inference_mode():
                result = self._model(
                    prompt=prompt,
                    negative_prompt=neg,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=gen,
                )
            image = result.images[0]
        finally:
            stop.set()
            if ticker:
                ticker.join(timeout=1.0)

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 98, "saving")

        if self.outputs_dir:
            out_dir = self.outputs_dir
        else:
            out_dir = self.model_dir.parent.parent.parent / "outputs" / self.MODEL_ID

        out_dir.mkdir(parents=True, exist_ok=True)
        filename = "hunyuandit_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
        out_path = out_dir / filename
        image.save(str(out_path), format="PNG")

        self._report(progress_cb, 100, "done")
        print("[HunyuanDiT] saved %s" % out_path)
        return str(out_path)

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        repo = self.hf_repo or HF_REPO
        skips = list(getattr(self, "hf_skip_prefixes", []) or [])

        ignore = []
        for p in skips:
            ignore.append(p)
            if isinstance(p, str) and p.endswith("/"):
                ignore.append(p + "*")
        ignore += ["*.md", "*.txt", "LICENSE", "NOTICE", ".gitattributes"]

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[HunyuanDiT] downloading from %s" % repo)
        snapshot_download(repo_id=repo, local_dir=str(self.model_dir), ignore_patterns=ignore)
        print("[HunyuanDiT] download complete")
