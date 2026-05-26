"""
HunyuanDiT Turbo – Text-to-Image generator for Modly.
"""
import io
import os
import base64
from pathlib import Path
from typing import Optional

# Diffusers-compatible HunyuanDiT Turbo (distilled) repo on HuggingFace.
HF_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"


def _resolve_models_dir():
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "ModlyData" / "models"


def _get_hf_token():
    for key in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    return None


class HunyuanT2IGenerator:
    """
    Modly generator class for Hunyuan DiT Turbo text-to-image.
    """

    def __init__(self, models_dir=None, **kwargs):
        self._models_dir = Path(models_dir).resolve() if models_dir else _resolve_models_dir()
        self._local_dir = self._models_dir / "hunyuan_t2i_turbo" / "generate"
        self._pipe = None

    # ------------------------------------------------------------------ #
    #  Download                                                            #
    # ------------------------------------------------------------------ #
    def download(self, repo_id=None, models_dir=None, **kwargs):
        """
        Called by Modly when the user presses the Download button in the UI.
        Downloads the full model snapshot from HuggingFace.
        """
        from huggingface_hub import snapshot_download

        if models_dir:
            self._models_dir = Path(models_dir).resolve()
            self._local_dir = self._models_dir / "hunyuan_t2i_turbo" / "generate"

        repo = (repo_id or HF_REPO).strip()
        self._local_dir.mkdir(parents=True, exist_ok=True)
        token = _get_hf_token()

        print("[HunyuanT2I] Downloading '{}' -> {}".format(repo, self._local_dir))
        snapshot_download(
            repo_id=repo,
            local_dir=str(self._local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        print("[HunyuanT2I] Download complete.")
        return {"status": "ok", "local_dir": str(self._local_dir)}

    # ------------------------------------------------------------------ #
    #  Load pipeline                                                       #
    # ------------------------------------------------------------------ #
    def _load_pipe(self, models_dir=None):
        import torch
        from diffusers import HunyuanDiTPipeline

        if models_dir:
            local_dir = Path(models_dir).resolve() / "hunyuan_t2i_turbo" / "generate"
        else:
            local_dir = self._local_dir

        print("[HunyuanT2I] Loading pipeline from {} ...".format(local_dir))
        self._pipe = HunyuanDiTPipeline.from_pretrained(
            str(local_dir),
            torch_dtype=torch.float16,
        )
        self._pipe.enable_model_cpu_offload()
        print("[HunyuanT2I] Pipeline ready.")

    # ------------------------------------------------------------------ #
    #  Run / Generate                                                      #
    # ------------------------------------------------------------------ #
    def run(self, inputs=None, params=None, models_dir=None, **kwargs):
        """
        Called by Modly for inference.
        inputs  : {"text": "a photo of a cat"}
        params  : {"num_inference_steps": 20, "seed": 42, "height": 1024, "width": 1024}
        """
        import torch

        inputs = inputs or {}
        params = params or {}

        prompt = inputs.get("text") or inputs.get("prompt", "")
        steps = int(params.get("num_inference_steps", 20))
        seed = int(params.get("seed", 42))
        height = int(params.get("height", 1024))
        width = int(params.get("width", 1024))

        if self._pipe is None:
            self._load_pipe(models_dir=models_dir)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=generator,
        )
        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_b64": b64}
