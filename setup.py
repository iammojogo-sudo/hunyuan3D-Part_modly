# Minimal Modly setup for a text-to-image Hunyuan T2I extension.
import json, os, platform, subprocess, sys
from pathlib import Path

IS_WIN = platform.system() == "Windows"

def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)

def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")

def setup(python_exe, ext_dir, gpu_sm):
    venv = Path(ext_dir) / "venv"
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    print("[setup] venv created:", venv)
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")
    # Install torch appropriate for GPU; keep simple fallback to cpu
    try:
        if gpu_sm >= 70:
            pip(venv, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124")
        else:
            pip(venv, "install", "torch", "torchvision")
    except Exception:
        pip(venv, "install", "torch", "torchvision")
    pip(venv, "install", "diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub", "Pillow")
    # No repo clone here; generator will snapshot_download on demand
    print("[setup] Done. venv at:", venv)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        setup(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        args = json.loads(sys.argv[1])
        setup(args["python_exe"], args["ext_dir"], int(args["gpu_sm"]))
