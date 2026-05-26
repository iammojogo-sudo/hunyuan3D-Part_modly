"""
Modly extension setup script.

Creates the extension venv and installs runtime packages. Keeps behavior minimal
so Modly can install the extension without manual steps.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4050)
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
                    "via the environment variable HUGGINGFACE_HUB_TOKEN or HF_TOKEN before running setup."
                ) from exc
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        # backoff
        time.sleep(2)
    raise RuntimeError(f"Failed to download snapshot after {attempts} attempts: {last_exc}")


def setup(python_exe: str, ext_dir: str, gpu_sm: int):
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

    print("[setup] Done. venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=sys.argv[2],
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=args["ext_dir"],
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
