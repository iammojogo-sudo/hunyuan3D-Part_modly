import json
import os
import platform
import subprocess
import sys
from pathlib import Path

IS_WIN = platform.system() == "Windows"


def pip(venv_py, *args):
    subprocess.run([str(venv_py), "-m", "pip"] + list(args), check=True)


def venv_python(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def torch_index(gpu_sm):
    # cu121 for anything ampere and up, cu118 for older stuff
    if gpu_sm >= 80:
        return "https://download.pytorch.org/whl/cu121"
    return "https://download.pytorch.org/whl/cu118"


def setup(python_exe, ext_dir, gpu_sm):
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"

    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    py = venv_python(venv)

    pip(py, "install", "--upgrade", "pip", "setuptools", "wheel")

    idx = torch_index(gpu_sm)
    print("[setup] torch index:", idx)
    pip(py, "install", "torch", "torchvision", "--index-url", idx)

    pip(py, "install", "--upgrade", "huggingface_hub>=0.16.4")

    pip(py, "install", "--upgrade",
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "tqdm",
    )

    print("[setup] done")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) == 2:
        a = json.loads(sys.argv[1])
        setup(a["python_exe"], a["ext_dir"], int(a["gpu_sm"]))
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
