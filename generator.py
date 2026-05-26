"""
Hunyuan T2I Turbo generator for Modly (single-file replacement).

- Module-level download entrypoints: hf_download, download_model, download
- Uses MODELS_DIR env var or fallback to ~/ModlyData/models
- Reads manifest.json to honor hf_skip_prefixes and download_check for the node
- Writes a timestamped marker file when invoked so you can confirm invocation
- Uses huggingface_hub.snapshot_download with optional auth token
- Minimal changes; keeps generation behavior intact
"""
import io
import os
import sys
import time
import uuid
import threading
from pathlib import Path
from typing import Optional, List, Dict

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress

# Default HF repo for this extension
_DEFAULT_HF_REPO = "TencentARC/HunyuanDiT-Turbo"
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
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    marker_name = f"hf_download_invoked_{ts}_{uuid.uuid4().hex[:8]}.txt"
    marker_path = ext_dir / marker_name
    try:
        marker_path.write_text(f"invoked: {time.asctime()}\nrepo_id: {repo_id}\n", encoding="utf-8")
    except Exception:
        pass
    return marker_path


def _load_manifest_ignore_patterns(ext_dir: Path, model_id: Optional[str]) -> List[str]:
    """
    Read manifest.json in extension root and return hf_skip_prefixes for the node
    whose id matches model_id. If not found, return a sensible default ignore list.
    """
    manifest_path = ext_dir / "manifest.json"
    ignore = ["*.md", "*.txt", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"]
    if not manifest_path.exists():
        return ignore
    try:
        import json
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        nodes = m.get("nodes", []) or []
        mid = None
        if model_id:
            # model_id may be URL-encoded or include node suffix; normalize
            mid = model_id.split("/")[-1]
            mid = mid.replace("%2F", "/")
        for node in nodes:
            nid = node.get("id", "")
            if nid and (nid == mid or (model_id and model_id.endswith("/" + nid))):
                skips = node.get("hf_skip_prefixes") or []
                if isinstance(skips, list):
                    for p in skips:
                        ignore.append(p)
                        if isinstance(p, str) and p.endswith("/"):
                            ignore.append(p + "*")
                dc = node.get("download_check")
                if isinstance(dc, str) and "/" in dc:
                    prefix = dc.split("/")[0] + "/"
                    ignore.append(prefix)
                    ignore.append(prefix + "*")
                break
    except Exception:
        pass
    return ignore


def _snapshot_download_with_token(repo_id: str, local_dir: str, token: Optional[str], ignore_patterns: Optional[List[str]] = None, attempts: int = 3):
    """
    Wrapper around huggingface_hub.snapshot_download with retries and token support.
    Raises RuntimeError on 401 or after attempts exhausted.
    """
    from huggingface_hub import snapshot_download
    from httpx import HTTPStatusError

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=ignore_patterns,
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
        time.sleep(2)
    raise RuntimeError(f"Failed to download snapshot after {attempts} attempts: {last_exc}")


def hf_download(repo_id: str, model_id: Optional[str] = None) -> Dict:
    """
    Module-level download function intended to be called by Modly's PythonBridge
    when the user presses the Download button.

    Returns:
        { "success": bool, "path": str|None, "message": str }
    """
    try:
        ext_dir = Path(__file__).parent.resolve()
    except Exception:
        ext_dir = Path.cwd()

    marker = _write_invocation_marker(ext_dir, repo_id or _DEFAULT_HF_REPO)
    print(f"[HunyuanT2IGenerator] hf_download invoked; marker written: {marker}", file=sys.stderr)

    repo = repo_id or _DEFAULT_HF_REPO
    token = _get_hf_token()
    target_dir = _repo_target_dir(repo)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    ignore = _load_manifest_ignore_patterns(ext_dir, model_id)
    try:
        _snapshot_download_with_token(repo, str(target_dir), token, ignore_patterns=ignore, attempts=_DOWNLOAD_ATTEMPTS)
        msg = f"Download complete: {target_dir}"
        print(f"[HunyuanT2IGenerator] {msg}", file=sys.stderr)
        return {"success": True, "path": str(target_dir), "message": msg}
    except Exception as exc:
        err = str(exc)
        print(f"[HunyuanT2IGenerator] Download error: {err}", file=sys.stderr)
        return {"success": False, "path": None, "message": err}


# Aliases for compatibility with different mappings Modly might use
def download_model(repo_id: str, model_id: Optional[str] = None) -> Dict:
    return hf_download(repo_id, model_id)


def download(repo_id: str, model_id: Optional[str] = None) -> Dict:
    return hf_download(repo_id, model_id)


# ----------------- Generator class -----------------
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
        """
        Class-level download used by generator runtime (same location as hf_download).
        Honors manifest hf_skip_prefixes if present on the node.
        """
        from huggingface_hub import snapshot_download
        from httpx import HTTPStatusError

        repo_id = self.hf_repo or _DEFAULT_HF_REPO
        token = _get_hf_token()

        # Build ignore list from manifest-provided hf_skip_prefixes if available
        manifest_skips = list(getattr(self, "hf_skip_prefixes", []) or [])
        ignore = []
        for pattern in manifest_skips:
            ignore.append(pattern)
            if isinstance(pattern, str) and pattern.endswith("/"):
                ignore.append(pattern + "*")
        ignore += ["*.md", "*.txt", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"]

        target_dir = _repo_target_dir(repo_id)
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        last_exc = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    ignore_patterns=ignore,
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

        try:
            from diffusers import DiffusionPipeline
            import torch
        except Exception as exc:
            raise RuntimeError("Required packages not installed in extension venv: %s" % exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        repo_dir = _repo_target_dir(self.hf_repo or _DEFAULT_HF_REPO)

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

        # Save output
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8]))
        image.save(str(out_path))
        self._report(progress_cb, 100, "Done")
        return str(out_path)
