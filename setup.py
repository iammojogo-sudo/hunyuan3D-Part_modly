"""
Modly extension setup script for Hunyuan T2I Turbo.

Creates a venv, installs runtime packages, and (optionally) pre-downloads
the Hugging Face repo using an auth token if required.

Usage (called by Modly):
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

IS_WIN = platform.system() == "Windows"

HF_REPO = "TencentARC/HunyuanDiT-Turbo"


def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _get_hf_token():
    # Accept common env var names
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _snapshot_download_with_token(repo_id, local_dir, token=None, attempts=3):
    from huggingface_hub import snapshot_download
    from httpx import HTTPStatusError

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
            if attempt < attempts:
                time.sleep(2)
                continue
            raise
        except Exception:
            if attempt < attempts:
                time.sleep(2)
                continue
            raise


def setup(python_exe, ext_dir, gpu_sm):
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)

    venv_python = python_exe_in_venv(venv)

    print("[setup] Installing build/runtime prerequisites (pip, setuptools, wheel)...")
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    # Install huggingface_hub first (used for snapshot_download)
    print("[setup] Installing huggingface_hub into venv (required for snapshot_download)...")
    pip(venv, "install", "--upgrade", "huggingface_hub>=0.16.4")

    # Install core runtime packages
    print("[setup] Installing core runtime dependencies into venv...")
    # Keep versions flexible but modern; adjust if you need pinned versions
    pip(venv, "install", "--upgrade",
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "tqdm",
    )

    # Optionally pre-download the HF repo during setup if token is available.
    # This speeds first-run but is not required; generator will also download on demand.
    token = _get_hf_token()
    if token:
        print("[setup] HF token detected; attempting to pre-download model repo %s ..." % HF_REPO)
        tmp_dir = Path(os.environ.get("TEMP", "/tmp")) / ("modly_hf_" + str(int(time.time())))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            _snapshot_download_with_token(HF_REPO, str(tmp_dir), token=token)
            # Move snapshot into extension model_dir for convenience
            model_dir = ext_dir / "model"
            if model_dir.exists():
                # merge by copying files (best-effort)
                for src in tmp_dir.iterdir():
                    dest = model_dir / src.name
                    if not dest.exists():
                        try:
                            if src.is_dir():
                                import shutil
                                shutil.copytree(src, dest)
                            else:
                                src.replace(dest)
                        except Exception:
                            pass
            else:
                try:
                    tmp_dir.replace(model_dir)
                except Exception:
                    # fallback: leave in tmp_dir; generator will snapshot_download again
                    pass
            print("[setup] Pre-download complete.")
        except Exception as exc:
            print("[setup] Pre-download failed: %s" % exc)
            print("[setup] Continuing setup; generator will attempt download at runtime.")
    else:
        print("[setup] No HF token found in environment; skipping pre-download. Provide HUGGINGFACE_HUB_TOKEN to pre-download during setup.")

    print("[setup] Done. Venv ready at: %s" % venv)


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
