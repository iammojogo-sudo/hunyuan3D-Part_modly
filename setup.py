"""
Modly extension setup script (download-action enabled).

This script creates the extension venv and installs runtime packages.
It also supports an explicit "download" action to fetch model weights into
Modly's shared models directory (used by the purple Download button).

Usage (called by Modly):
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4050)
    action      - optional string: "install" (default) or "download"

Alternatively, call:
    python setup.py download <repo_id>

Environment:
    MODELS_DIR - optional path to Modly's models directory (preferred).
                 If not set, fallback to: ~/ModlyData/models
    HUGGINGFACE_HUB_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN - optional HF token for gated repos.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

IS_WIN = platform.system() == "Windows"

# Default HF repo used for optional pre-download. Change if you want a different repo.
DEFAULT_HF_REPO = "TencentARC/HunyuanDiT-Turbo"


def run_venv_pip(venv_python: Path, *pip_args, check=True):
    """
    Run the venv Python with -m pip so pip upgrades/installations are robust on Windows.
    """
    cmd = [str(venv_python), "-m", "pip"] + list(pip_args)
    subprocess.run(cmd, check=check)


def python_exe_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _get_hf_token() -> Optional[str]:
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _resolve_models_dir() -> Path:
    """
    Resolve the shared models directory. Prefer MODELS_DIR env var (set by Modly).
    Fallback to a safe user-home location: ~/ModlyData/models
    """
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # fallback
    return Path.home() / "ModlyData" / "models"


def _snapshot_download_with_token(repo_id: str, local_dir: str, token: Optional[str], attempts: int = 3):
    """
    Use huggingface_hub.snapshot_download with retries and explicit token passing.
    Raises RuntimeError with a clear message on 401 Unauthorized.
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
        # backoff
        time.sleep(2)
    raise RuntimeError(f"Failed to download snapshot after {attempts} attempts: {last_exc}")


def setup_install(python_exe: str, ext_dir: str, gpu_sm: int):
    """
    Create venv and install runtime packages. Does not force model download unless
    an HF token is present and pre-download is desired.
    """
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)

    venv_python = python_exe_in_venv(venv)

    print("[setup] Ensuring pip/setuptools/wheel are up-to-date in venv...")
    run_venv_pip(venv_python, "install", "--upgrade", "pip", "setuptools", "wheel")

    print("[setup] Installing huggingface_hub into venv (required for snapshot_download)...")
    run_venv_pip(venv_python, "install", "--upgrade", "huggingface_hub>=0.16.4")

    print("[setup] Installing core runtime dependencies into venv...")
    core_pkgs = [
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "tqdm",
    ]
    run_venv_pip(venv_python, "install", "--upgrade", *core_pkgs)

    print("[setup] Setup complete. venv ready at: %s" % venv)


def setup_download(repo_id: str):
    """
    Download the HF repo into the shared models directory.
    This is the action the Modly 'Download' button should trigger.
    """
    models_dir = _resolve_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Place each repo into a subfolder named after the repo id (safe)
    safe_name = repo_id.replace("/", "_")
    target_dir = models_dir / safe_name

    token = _get_hf_token()
    print(f"[setup] Downloading repo '{repo_id}' into models dir: {models_dir} (auth={'yes' if token else 'no'})")
    try:
        _snapshot_download_with_token(repo_id, str(target_dir), token=token)
        print(f"[setup] Download complete: {target_dir}")
    except Exception as exc:
        # Raise so Modly can surface the error to the user
        raise


def main_from_json_arg(json_arg: str):
    """
    Accept a JSON string argument from Modly with keys:
      python_exe, ext_dir, gpu_sm, action (optional), repo_id (optional)
    """
    args = json.loads(json_arg)
    python_exe = args.get("python_exe")
    ext_dir = args.get("ext_dir")
    gpu_sm = int(args.get("gpu_sm", 0))
    action = args.get("action", "install")
    repo_id = args.get("repo_id", DEFAULT_HF_REPO)

    if action == "install":
        setup_install(python_exe, ext_dir, gpu_sm)
    elif action == "download":
        setup_download(repo_id)
    else:
        raise RuntimeError(f"Unknown action: {action}")


if __name__ == "__main__":
    # CLI modes:
    # 1) python setup.py <json_args>
    # 2) python setup.py download <repo_id>
    # 3) python setup.py install <python_exe> <ext_dir> <gpu_sm>
    if len(sys.argv) == 2:
        # assume JSON arg
        main_from_json_arg(sys.argv[1])
    elif len(sys.argv) >= 2 and sys.argv[1] == "download":
        repo = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_HF_REPO
        setup_download(repo)
    elif len(sys.argv) == 4 and sys.argv[1] == "install":
        # python setup.py install <python_exe> <ext_dir> <gpu_sm>
        setup_install(sys.argv[2], sys.argv[3], int(0))
    else:
        print("Usage:")
        print("  python setup.py '<json_args>'")
        print("  python setup.py download <repo_id>")
        print("  python setup.py install <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
