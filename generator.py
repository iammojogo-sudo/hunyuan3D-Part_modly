"""
HunyuanDiT Turbo - Text-to-Image generator for Modly.

Node layout
-----------
generate  : text  -> image   (T2I diffusion)
preview   : image -> image   (passthrough display)
save      : image -> image   (OS save-file dialog)
"""
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------- #
#  Ensure the extension's own venv is on sys.path regardless of which    #
#  Python Modly uses to import this file.                                 #
# ---------------------------------------------------------------------- #
_EXT_DIR = Path(__file__).parent
for _venv in (_EXT_DIR / "venv", _EXT_DIR / ".venv"):
    if _venv.exists():
        # Windows
        _win_sp = _venv / "Lib" / "site-packages"
        if _win_sp.exists() and str(_win_sp) not in sys.path:
            sys.path.insert(0, str(_win_sp))
        # Linux / macOS (python3.x subfolder)
        for _lib in (_venv / "lib").glob("python3.*"):
            _unix_sp = _lib / "site-packages"
            if _unix_sp.exists() and str(_unix_sp) not in sys.path:
                sys.path.insert(0, str(_unix_sp))
        break

HF_REPO   = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"
_TEMP_DIR = Path(tempfile.gettempdir()) / "hunyuan_t2i_turbo"


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


def _find_model_dir(base):
    """
    Search for the directory that contains model_index.json, trying several
    path layouts Modly may use depending on its version.
    """
    base = Path(base)
    candidates = [
        base / "hunyuan_t2i_turbo" / "generate",
        base / "generate",
        base,
    ]
    for c in candidates:
        if (c / "model_index.json").exists():
            return c
    # Fallback: recursive search
    for p in base.rglob("model_index.json"):
        return p.parent
    # Nothing found - return the most likely path so the error message is useful
    return candidates[0]


def _extract_image_path(inputs):
    if isinstance(inputs, str):
        return inputs.strip()
    if isinstance(inputs, dict):
        return str(inputs.get("image") or inputs.get("image_path") or "").strip()
    return ""


class HunyuanT2IGenerator:
    """Modly generator class for Hunyuan DiT Turbo text-to-image."""

    def __init__(self, models_dir=None, **kwargs):
        self._models_dir = (
            Path(models_dir).resolve() if models_dir else _resolve_models_dir()
        )
        self._pipe = None
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print("[HunyuanT2I] Extension loaded. models_dir={}".format(self._models_dir))

    # ------------------------------------------------------------------ #
    #  run() dispatcher (Modly may call this instead of named methods)    #
    # ------------------------------------------------------------------ #
    def run(self, node_id=None, inputs=None, params=None, models_dir=None, **kwargs):
        inputs = inputs or {}
        params = params or {}
        if node_id == "preview":
            return self.preview(inputs=inputs, params=params)
        if node_id == "save":
            return self.save(inputs=inputs, params=params)
        return self.generate(inputs=inputs, params=params, models_dir=models_dir)

    # ------------------------------------------------------------------ #
    #  Download                                                            #
    # ------------------------------------------------------------------ #
    def download(self, repo_id=None, models_dir=None, **kwargs):
        from huggingface_hub import snapshot_download

        if models_dir:
            self._models_dir = Path(models_dir).resolve()

        local_dir = self._models_dir / "hunyuan_t2i_turbo" / "generate"
        repo      = (repo_id or HF_REPO).strip()
        token     = _get_hf_token()
        local_dir.mkdir(parents=True, exist_ok=True)

        print("[HunyuanT2I] Downloading '{}' -> {}".format(repo, local_dir))
        snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        print("[HunyuanT2I] Download complete.")
        return {"status": "ok", "local_dir": str(local_dir)}

    # ------------------------------------------------------------------ #
    #  Load pipeline                                                       #
    # ------------------------------------------------------------------ #
    def _load_pipe(self, models_dir=None):
        import torch
        from diffusers import HunyuanDiTPipeline

        search_base = Path(models_dir).resolve() if models_dir else self._models_dir
        local_dir   = _find_model_dir(search_base)

        print("[HunyuanT2I] Resolved model dir: {}".format(local_dir))

        if not local_dir.exists():
            raise RuntimeError(
                "[HunyuanT2I] Model weights not found at '{}'.\n"
                "Please open the Extensions panel and click Download first.".format(local_dir)
            )

        print("[HunyuanT2I] Loading pipeline ...")
        self._pipe = HunyuanDiTPipeline.from_pretrained(
            str(local_dir),
            torch_dtype=torch.float16,
        )
        self._pipe.enable_model_cpu_offload()
        print("[HunyuanT2I] Pipeline ready.")

    # ------------------------------------------------------------------ #
    #  NODE: generate  (text -> image)                                     #
    # ------------------------------------------------------------------ #
    def generate(self, inputs=None, params=None, models_dir=None, **kwargs):
        print("[HunyuanT2I] generate() called")
        import torch

        inputs = inputs or {}
        params = params or {}

        prompt = inputs.get("text") or inputs.get("prompt", "")
        steps  = int(params.get("num_inference_steps", 20))
        seed   = int(params.get("seed", 42))
        height = int(params.get("height", 1024))
        width  = int(params.get("width", 1024))

        print("[HunyuanT2I] prompt='{}' steps={} seed={} {}x{}".format(
            prompt, steps, seed, width, height))

        if self._pipe is None:
            self._load_pipe(models_dir=models_dir)

        gen    = torch.Generator(device="cuda").manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=gen,
        )
        image    = result.images[0]
        out_path = _TEMP_DIR / "output_{}.png".format(seed)
        image.save(str(out_path), format="PNG")
        print("[HunyuanT2I] Saved -> {}".format(out_path))
        return {"image": str(out_path)}

    # Alias in case Modly maps node name rather than node id
    generate_image = generate

    # ------------------------------------------------------------------ #
    #  NODE: preview  (image -> image passthrough)                         #
    # ------------------------------------------------------------------ #
    def preview(self, inputs=None, params=None, **kwargs):
        print("[HunyuanT2I] preview() called")
        inputs     = inputs or {}
        image_path = _extract_image_path(inputs)
        if not image_path or not Path(image_path).exists():
            raise FileNotFoundError(
                "[HunyuanT2I] Preview node got no valid image path (got '{}').".format(image_path)
            )
        return {"image": image_path}

    # ------------------------------------------------------------------ #
    #  NODE: save  (image -> image, opens OS save dialog)                  #
    # ------------------------------------------------------------------ #
    def save(self, inputs=None, params=None, **kwargs):
        print("[HunyuanT2I] save() called")
        import tkinter as tk
        from tkinter import filedialog
        from PIL import Image

        inputs     = inputs or {}
        params     = params or {}
        image_path = _extract_image_path(inputs)

        if not image_path or not Path(image_path).exists():
            raise FileNotFoundError(
                "[HunyuanT2I] Save node got no valid image path (got '{}').".format(image_path)
            )

        fmt = params.get("format", "PNG").strip().upper()
        if fmt == "JPG":
            fmt = "JPEG"

        ext_map   = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
        ext       = ext_map.get(fmt, ".png")
        filetypes = [
            ("PNG image",  "*.png"),
            ("JPEG image", "*.jpg"),
            ("WebP image", "*.webp"),
            ("All files",  "*.*"),
        ]

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        save_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=ext,
            filetypes=filetypes,
        )
        root.destroy()

        if not save_path:
            return {"image": image_path}

        img = Image.open(image_path)
        if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            if img.mode == "P":
                img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg

        img.save(save_path, format=fmt)
        print("[HunyuanT2I] Saved -> {}".format(save_path))
        return {"image": save_path}
