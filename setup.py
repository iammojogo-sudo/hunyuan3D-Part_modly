"""
Modly extension setup script (fixed).

This version avoids invoking the venv pip executable directly (which can fail
when upgrading pip on Windows). Instead it runs the venv Python with `-m pip`
for all package installs and upgrades. It also handles authenticated HF
snapshot downloads (when HUGGINGFACE_HUB_TOKEN / HF_TOKEN is present).

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

IS_WIN = platform.system() == "Windows"

# Default HF repo used for optional pre-download. Change if you want a different repo.
HF_REPO = "TencentARC/HunyuanDiT-Turbo"


def run_venv_pip(venv_python: Path, *pip_args, check=True):
    """
    Run the venv Python with -m pip so pip upgrades/installations are robust on Windows.
    Example: run_venv_pip(venv_python, "install", "--upgrade", "pip")
    """
    cmd = [str(venv_python), "-m", "pip"] + list(pip_args)
    subprocess.run(cmd, check=check)


def python_exe_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _get_hf_token():
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _snapshot_download_with_token(repo_id: str, local_dir: str, token: str | None, attempts: int = 3):
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
    # Use python -m pip to avoid pip.exe self-upgrade issues on Windows
    run_venv_pip(venv_python, "install", "--upgrade", "pip", "setuptools", "wheel")

    # Install huggingface_hub first (used for snapshot_download)
    print("[setup] Installing huggingface_hub into venv (required for snapshot_download)...")
    run_venv_pip(venv_python, "install", "--upgrade", "huggingface_hub>=0.16.4")

    # Install core runtime packages
    print("[setup] Installing core runtime dependencies into venv...")
    # Keep versions flexible; pin if you need reproducible installs
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
                import shutil

                for src in tmp_dir.iterdir():
                    dest = model_dir / src.name
                    if not dest.exists():
                        try:
                            if src.is_dir():
                                shutil.copytree(src, dest)
                            else:
                                shutil.copy2(src, dest)
                        except Exception:
                            # ignore individual copy errors
                            pass
            else:
                try:
                    tmp_dir.replace(model_dir)
                except Exception:
                    # fallback: try copytree
                    try:
                        import shutil

                        shutil.copytree(tmp_dir, model_dir)
                    except Exception:
                        pass
            print("[setup] Pre-download complete.")
        except Exception as exc:
            # If pre-download fails, do not abort the whole setup; generator will attempt download at runtime.
            print("[setup] Pre-download failed: %s" % exc)
            print("[setup] Continuing setup; generator will attempt download at runtime.")
    else:
        print("[setup] No HF token found in environment; skipping pre-download. Provide HUGGINGFACE_HUB_TOKEN to pre-download during setup.")

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
