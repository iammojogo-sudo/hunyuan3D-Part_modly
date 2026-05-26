"""
Modly extension setup script.

Creates the extension venv and installs runtime packages.

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
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "ModlyData" / "models"


def _torch_index_url(gpu_sm: int) -> str:
    """
    Return the correct PyTorch wheel index URL for the detected GPU.
    gpu_sm is the CUDA compute capability as an integer (e.g. 89 for SM 8.9).
    SM >= 80 (Ampere / Ada / Hopper) -> CUDA 12.1
    SM  < 80 (Turing / Volta / older) -> CUDA 11.8
    """
    if gpu_sm >= 80:
        return "https://download.pytorch.org/whl/cu121"
    return "https://download.pytorch.org/whl/cu118"


def setup(python_exe: str, ext_dir: str, gpu_sm: int):
    ext_dir = Path(ext_dir)
    venv    = ext_dir / "venv"

    print("[setup] Creating venv at {} ...".format(venv))
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)

    venv_python = python_exe_in_venv(venv)

    print("[setup] Upgrading pip / setuptools / wheel ...")
    run_venv_pip(venv_python, "install", "--upgrade", "pip", "setuptools", "wheel")

    torch_index = _torch_index_url(gpu_sm)
    print("[setup] Installing PyTorch (index: {}) ...".format(torch_index))
    run_venv_pip(
        venv_python,
        "install",
        "torch",
        "torchvision",
        "--index-url", torch_index,
    )

    print("[setup] Installing huggingface_hub ...")
    run_venv_pip(venv_python, "install", "--upgrade", "huggingface_hub>=0.16.4")

    print("[setup] Installing core runtime dependencies ...")
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

    print("[setup] Done. venv ready at: {}".format(venv))


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
