"""
Hunyuan3D-Part — Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4090)
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


IS_WIN = platform.system() == "Windows"


def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    if not venv.exists():
        print("[setup] Creating venv at %s ..." % venv)
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("[setup] Venv exists, skipping creation.")

    venv_python = python_exe_in_venv(venv)

    # ------------------------------------------------------------------ #
    # Build prerequisites
    # ------------------------------------------------------------------ #
    print("[setup] Installing build prerequisites...")
    pip(venv, "install", "ninja", "setuptools", "wheel")

    # ------------------------------------------------------------------ #
    # PyTorch — same arch tiers as the 2mv extension
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
        print("[setup] SM %d -> PyTorch 2.6.0 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # xformers (non-fatal — used for extra attention kernels)
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers (non-fatal)...")
    try:
        if gpu_sm >= 70:
            pip(venv, "install", "xformers==0.0.29.post3", "--index-url", torch_index)
        else:
            pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
                "https://download.pytorch.org/whl/cu118")
    except subprocess.CalledProcessError:
        print("[setup] xformers install failed — skipping (non-fatal).")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    pip(venv, "install",
        "accelerate",
        "einops",
        "omegaconf",
        "timm",
        "diffusers",
        "Pillow",
        "numpy",
        "scipy",
        "scikit-image",
        "trimesh",
        "pymeshlab",
        "tqdm",
        "safetensors",
        "huggingface_hub",
        "psutil",
    )

    # triton: Linux-only
    if not IS_WIN:
        try:
            pip(venv, "install", "triton")
        except subprocess.CalledProcessError:
            print("[setup] triton not available — skipping (non-fatal).")

    # ------------------------------------------------------------------ #
    # onnxruntime
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        print("[setup] Installing onnxruntime-gpu...")
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            print("[setup] onnxruntime-gpu failed, falling back to cpu.")
            pip(venv, "install", "onnxruntime")
    else:
        pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Clone Hunyuan3D-Part repo (code only — weights download at first run)
    # ------------------------------------------------------------------ #
    repo_dir = ext_dir / "Hunyuan3D-Part"
    if not repo_dir.exists():
        print("[setup] Cloning Hunyuan3D-Part repo...")
        cloned = False
        for url in [
            "https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git",
            "https://github.com/tencent/Hunyuan3D-Part.git",
        ]:
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", url, str(repo_dir)],
                    check=True,
                )
                cloned = True
                print("[setup] Cloned from %s" % url)
                break
            except subprocess.CalledProcessError:
                print("[setup] Clone from %s failed, trying next..." % url)
        if not cloned:
            print(
                "[setup] WARNING: Could not clone Hunyuan3D-Part repo.\n"
                "[setup]   Download manually from https://github.com/Tencent-Hunyuan/Hunyuan3D-Part\n"
                "[setup]   and place it at: %s" % repo_dir
            )
    else:
        print("[setup] Repo already exists, skipping clone.")

    # ------------------------------------------------------------------ #
    # Install partgen package from XPart/ if it has a setup file
    # ------------------------------------------------------------------ #
    xpart_dir = repo_dir / "XPart"
    if xpart_dir.exists():
        has_setup = (xpart_dir / "setup.py").exists() or (xpart_dir / "pyproject.toml").exists()
        if has_setup:
            print("[setup] Installing partgen package from XPart/...")
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-e", str(xpart_dir)],
                check=False,
            )
        else:
            print("[setup] No setup.py/pyproject.toml in XPart/ — will use sys.path at runtime.")
    else:
        print("[setup] XPart/ not found in repo — will check at runtime.")

    # ------------------------------------------------------------------ #
    # Verify torch import
    # ------------------------------------------------------------------ #
    print("[setup] Verifying torch import...")
    check = subprocess.run(
        [str(venv_python), "-c", "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print("[setup] %s" % check.stdout.strip())
    else:
        print("[setup] WARNING: torch import check failed:\n%s" % check.stderr.strip())

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=Path(sys.argv[1]),
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=Path(args["python_exe"]),
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
