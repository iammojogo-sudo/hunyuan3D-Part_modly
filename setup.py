import json
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
    # cu121 for Ampere+ (SM >= 80), cu118 for older GPUs
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

    pip(py, "install",
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

    # xformers is optional; try to install but don't fail the setup if it isn't available
    try:
        pip(py, "install", "xformers")
    except Exception:
        print("[setup] xformers install failed or not available; continuing without it")

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
