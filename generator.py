"""
Hunyuan T2I Turbo generator for Modly.

Single-file replacement that:
 - Exposes top-level download entrypoints: hf_download, download_model, download
 - Writes a timestamped marker file into the extension folder when invoked
 - Downloads the HF repo into Modly's shared models directory (MODELS_DIR or ~/ModlyData/models)
 - Loads the Diffusers pipeline from the shared models directory for generation

Notes:
 - No additional files required.
 - For gated repos, Modly must inject HUGGINGFACE_HUB_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN
   into the extension process environment so snapshot_download can authenticate.
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


def _write_invocation_marker(ext_dir: Path, repo_id: str) -> Path:
    """
    Write a small marker file into the extension directory so we can confirm
    the download entrypoint was invoked by Modly.
    """
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    marker_name = f"hf_download_invoked_{ts}_{uuid.uuid4().hex[:8]}.txt"
    marker_path = ext_dir / marker_name
    try:
        marker_path.write_text(f"invoked: {time.asctime()}\nrepo_id: {repo_id}\n", encoding="utf-8")
    except Exception:
        # best-effort; do not fail the download just because marker couldn't be written
        pass
    return marker_path


def _snapshot_download(repo_id: str, target_dir: str, token: Optional[str]):
    """
    Wrapper around huggingface_hub.snapshot_download with retries and clear 401 handling.
    """
    from huggingface_hub import snapshot_download
    from httpx import HTTPStatusError

    last_exc = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
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
                    "via the environment variable HUGGINGFACE_HUB_TOKEN or HF_TOKEN."
                ) from exc
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        # backoff
        time.sleep(2)
    raise RuntimeError(f"Failed to download snapshot after {_DOWNLOAD_ATTEMPTS} attempts: {last_exc}")


def _perform_download(repo_id: str, model_id: Optional[str] = None) -> dict:
    """
    Core download logic used by all top-level entrypoints.
    Returns a dict: {success: bool, path: str|None, message: str}
    """
    try:
        ext_dir = Path(__file__).parent.resolve()
    except Exception:
        ext_dir = Path.cwd()

    # Write marker immediately so we can see invocation even if download fails
    marker = _write_invocation_marker(ext_dir, repo_id)
    print(f"[generator.py] hf_download invoked; marker written: {marker}", file=sys.stderr)

    token = _get_hf_token()
    target_dir = _repo_target_dir(repo_id)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        _snapshot_download(repo_id, str(target_dir), token)
        msg = f"Download complete: {target_dir}"
        print(f"[generator.py] {msg}", file=sys.stderr)
        return {"success": True, "path": str(target_dir), "message": msg}
    except Exception as exc:
        err = str(exc)
        print(f"[generator.py] Download error: {err}", file=sys.stderr)
        return {"success": False, "path": None, "message": err}


# --- Top-level entrypoints Modly might call ---------------------------------
def hf_download(repo_id: str, model_id: Optional[str] = None) -> dict:
    """
    Primary entrypoint expected by Modly's /model/hf-download route.
    """
    return _perform_download(repo_id or DEFAULT_HF_REPO, model_id)


def download_model(repo_id: str, model_id: Optional[str] = None) -> dict:
    """
    Alias entrypoint in case Modly maps to a different function name.
    """
    return _perform_download(repo_id or DEFAULT_HF_REPO, model_id)


def download(repo_id: str, model_id: Optional[str] = None) -> dict:
    """
    Another alias for compatibility.
    """
    return _perform_download(repo_id or DEFAULT_HF_REPO, model_id)


# --- Generator class (unchanged behavior, uses same shared models dir) ------
class HunyuanT2IGenerator(BaseGenerator):
    MODEL_ID = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB = 6
    MODEL_VARIANT = "t2i-turbo"

    def is_downloaded(self) -> bool:
        repo_dir = _repo_target_dir(self.hf_repo or DEFAULT_HF_REPO)
        marker = repo_dir / "model_index.json"
        return marker.exists()

    def _download_weights(self):
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

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8]))
        image.save(str(out_path))
        self._report(progress_cb, 100, "Done")
        return str(out_path)
