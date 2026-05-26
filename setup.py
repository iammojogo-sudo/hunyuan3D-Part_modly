#!/usr/bin/env python3
"""
setup.py — Install/uninstall entrypoint for HunyuanDiT-Turbo Modly extension.

Supported invocations:
  python setup.py '{"python_exe":"<path>","ext_dir":"<path>","gpu_sm":89}'
  python setup.py <python_exe> <ext_dir> <gpu_sm>
  python setup.py install
  python setup.py uninstall

Behavior:
  - Creates a venv inside the extension folder (ext_dir/venv) using the provided python_exe.
  - Installs build prerequisites (ninja, setuptools, wheel) into the venv.
  - Installs PyTorch appropriate for gpu_sm and xformers.
  - Installs core Python deps (diffusers, transformers, huggingface_hub, safetensors, accelerate, pillow, numpy, etc.)
  - Downloads the HF repo TencentARC/HunyuanDiT-Turbo into models/<ext_id>/ using huggingface_hub.snapshot_download.
  - Idempotent where possible and prints progress messages.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

EXT_ID = "hunyuan_t2i_turbo_modly"
HF_REPO = "TencentARC/HunyuanDiT-Turbo"
DOWNLOAD_CHECK = "config.json"  # file expected somewhere in the HF repo
DEPS_MARKER = ".deps_installed_v1"

IS_WIN = sys.platform.startswith("win")

def log(msg: str):
    print(f"[{EXT_ID}] {msg}", flush=True)

def _root(ext_dir: Optional[str] = None) -> Path:
    if ext_dir:
        return Path(ext_dir).resolve()
    return Path(__file__).parent.resolve()

def _venv_dir(root: Path) -> Path:
    return root / "venv"

def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")

def _venv_pip(venv: Path) -> Path:
    return venv / ("Scripts" if IS_WIN else "bin") / ("pip.exe" if IS_WIN else "pip")

def _run(cmd, env=None, cwd: Optional[Path] = None, check=True):
    log("Running: " + " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), env=env, cwd=(str(cwd) if cwd else None), check=check)

def _create_venv(python_exe: str, venv_dir: Path):
    if venv_dir.exists():
        log(f"Venv already exists at {venv_dir}; skipping creation.")
        return
    log(f"Creating venv at {venv_dir} using {python_exe}")
    _run([python_exe, "-m", "venv", str(venv_dir)])

def _pip_install_in_venv(venv: Path, packages):
    pip_exe = str(_venv_pip(venv))
    cmd = [pip_exe, "install", "--upgrade"] + packages
    _run(cmd)

def _verify_downloaded_model(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.name == DOWNLOAD_CHECK:
            return True
    return False

def _snapshot_download(tmp_dir: Path):
    # Use the venv python's -m huggingface_hub.snapshot_download when called from install flow
    # This helper is used by install() below where we call the venv python explicitly.
    raise RuntimeError("_snapshot_download should not be called directly")

def install(python_exe: str, ext_dir: str, gpu_sm: int):
    root = _root(ext_dir)
    venv_dir = _venv_dir(root)
    model_dir = root / "models" / EXT_ID
    marker = root / DEPS_MARKER

    # 1) Create venv
    _create_venv(python_exe, venv_dir)
    venv_python = str(_venv_python(venv_dir))
    venv_pip = str(_venv_pip(venv_dir))

    # 2) Install build prerequisites into venv
    log("Installing build prerequisites (ninja, setuptools, wheel) into venv...")
    _pip_install_in_venv(venv_dir, ["ninja", "setuptools", "wheel"])

    # 3) Install PyTorch appropriate for gpu_sm
    # NOTE: adjust versions as needed for your environment. This is conservative and widely compatible.
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        log(f"SM {gpu_sm} -> PyTorch 2.7+CUDA12.8")
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
        log(f"SM {gpu_sm} -> PyTorch 2.6.0+CUDA12.4")
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        log(f"SM {gpu_sm} (legacy) -> PyTorch 2.5.1+CUDA11.8")

    log("Installing PyTorch into venv (this may take a while)...")
    try:
        # pip accepts --index-url as a separate arg; pass it at the end
        _run([venv_pip, "install", "--upgrade"] + torch_pkgs + ["--index-url", torch_index])
    except subprocess.CalledProcessError as e:
        log(f"PyTorch install failed: {e}. You may need to install a compatible wheel manually.")
        raise

    # 4) Install xformers (best-effort)
    log("Installing xformers (best-effort)...")
    try:
        if gpu_sm >= 70:
            _run([venv_pip, "install", "xformers==0.0.29.post3", "--index-url", torch_index])
        else:
            _run([venv_pip, "install", "xformers==0.0.28.post2", "--index-url", "https://download.pytorch.org/whl/cu118"])
    except subprocess.CalledProcessError:
        log("xformers install failed — continuing without it (non-fatal).")

    # 5) Install core Python deps (diffusers, transformers, huggingface_hub, safetensors, accelerate, pillow, numpy, etc.)
    core_deps = [
        "huggingface_hub>=0.16.4",
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "scipy",
        "tqdm",
    ]
    log("Installing core Python dependencies into venv...")
    _pip_install_in_venv(venv_dir, core_deps)

    # 6) Create models dir and download HF repo if needed
    model_dir_parent = root / "models"
    model_dir_parent.mkdir(parents=True, exist_ok=True)

    if model_dir.exists() and _verify_downloaded_model(model_dir):
        log(f"Model already present at {model_dir}; skipping download.")
    else:
        tmp = Path(tempfile.mkdtemp(prefix=f"{EXT_ID}_hf_"))
        try:
            log(f"Downloading Hugging Face repo {HF_REPO} into temporary dir {tmp} ...")
            # Use the venv python to run snapshot_download so it uses the venv's huggingface_hub
            _run([venv_python, "-m", "huggingface_hub.snapshot_download",
                  "--repo_id", HF_REPO,
                  "--local_dir", str(tmp),
                  "--local_dir_use_symlinks", "False"])
            if not _verify_downloaded_model(tmp):
                log(f"Warning: {DOWNLOAD_CHECK} not found in downloaded repo. Proceeding but verify contents.")
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            shutil.move(str(tmp), str(model_dir))
            log(f"Model downloaded and moved into place: {model_dir}")
        except subprocess.CalledProcessError as e:
            log(f"Error during snapshot_download: {e}")
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            raise
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    # 7) Create deps marker for idempotency
    try:
        marker = root / DEPS_MARKER
        if not marker.exists():
            marker.write_text(json.dumps({"installed_by": EXT_ID, "python": venv_python}), encoding="utf-8")
            log(f"Created deps marker at {marker}")
        else:
            log("Deps marker already present.")
    except Exception as e:
        log(f"Warning: could not write deps marker: {e}")

    log("Install complete.")

def uninstall(ext_dir: str):
    root = _root(ext_dir)
    model_dir = root / "models" / EXT_ID
    marker = root / DEPS_MARKER
    venv_dir = _venv_dir(root)

    if model_dir.exists():
        log(f"Removing model directory {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        log("No model directory found to remove.")

    if venv_dir.exists():
        log(f"Removing venv directory {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)
    else:
        log("No venv directory found to remove.")

    if marker.exists():
        try:
            marker.unlink()
            log(f"Removed deps marker {marker}")
        except Exception as e:
            log(f"Failed to remove marker: {e}")

    log("Uninstall complete.")

def main():
    # Support JSON arg form
    if len(sys.argv) == 2 and sys.argv[1].startswith("{"):
        try:
            args = json.loads(sys.argv[1])
            python_exe = args.get("python_exe", sys.executable)
            ext_dir = args.get("ext_dir", str(Path(__file__).parent))
            gpu_sm = int(args.get("gpu_sm", 70))
            install(python_exe, ext_dir, gpu_sm)
            return
        except Exception as exc:
            log(f"Invalid JSON args: {exc}")
            sys.exit(2)

    # Positional form: python setup.py <python_exe> <ext_dir> <gpu_sm>
    if len(sys.argv) >= 4:
        python_exe = sys.argv[1]
        ext_dir = sys.argv[2]
        try:
            gpu_sm = int(sys.argv[3])
        except Exception:
            gpu_sm = 70
        install(python_exe, ext_dir, gpu_sm)
        return

    # Simple commands
    if len(sys.argv) == 2:
        cmd = sys.argv[1].lower()
        if cmd == "install":
            install(sys.executable, str(Path(__file__).parent), 70)
            return
        if cmd == "uninstall":
            uninstall(str(Path(__file__).parent))
            return

    print("Usage:")
    print("  python setup.py '{\"python_exe\":\"...\",\"ext_dir\":\"...\",\"gpu_sm\":89}'")
    print("  python setup.py <python_exe> <ext_dir> <gpu_sm>")
    print("  python setup.py install")
    print("  python setup.py uninstall")
    sys.exit(1)

if __name__ == "__main__":
    main()
