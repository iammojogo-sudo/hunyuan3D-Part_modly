import json
import platform
import subprocess
import sys
from pathlib import Path

IS_WIN = platform.system() == "Windows"


def pip(venv, *args):
    """
    Run pip inside the venv. venv is a Path to the venv folder.
    Example: pip(venv, 'install', 'package')
    """
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def torch_index(gpu_sm):
    if gpu_sm >= 100:
        return "https://download.pytorch.org/whl/cu128"
    if gpu_sm >= 70:
        return "https://download.pytorch.org/whl/cu124"
    return "https://download.pytorch.org/whl/cu118"


def setup(python_exe, ext_dir, gpu_sm):
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"
    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    venv_python = python_exe_in_venv(venv)

    # Upgrade pip/setuptools/wheel inside venv first
    print("[setup] Upgrading pip, setuptools, wheel in venv...")
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    # Install torch (index chosen by GPU SM)
    idx = torch_index(gpu_sm)
    print("[setup] torch index:", idx)
    pip(venv, "install", "torch", "torchvision", "--index-url", idx)

    # Core deps
    pip(
        venv,
        "install",
        "huggingface_hub>=0.16.4",
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "numpy",
        "tqdm",
    )

    # Optional xformers
    try:
        pip(venv, "install", "xformers")
    except Exception:
        print("[setup] xformers install failed or not available; continuing without it")

    print("[setup] done. venv at %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) == 2:
        a = json.loads(sys.argv[1])
        setup(a["python_exe"], a["ext_dir"], int(a["gpu_sm"]))
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
