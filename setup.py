#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

IS_WIN = platform.system() == "Windows"


def run_cmd(cmd: List[str], env=None, cwd: Path = None) -> subprocess.CompletedProcess:
    """
    Run a command and return CompletedProcess. Raises CalledProcessError on non-zero exit.
    """
    print(f"[setup] RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(cwd) if cwd else None)


def pip_with_venv_python(venv_python: Path, *args: str) -> None:
    """
    Run: <venv_python> -m pip <args...>
    This avoids invoking pip.exe directly while upgrading pip itself.
    """
    cmd = [str(venv_python), "-m", "pip"] + list(args)
    try:
        cp = run_cmd(cmd)
        print(cp.stdout)
        if cp.stderr:
            print(cp.stderr)
    except subprocess.CalledProcessError as e:
        print("[setup] ERROR: pip command failed.")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise


def python_exe_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def torch_index(gpu_sm: int) -> str:
    if gpu_sm >= 100:
        return "https://download.pytorch.org/whl/cu128"
    if gpu_sm >= 70:
        return "https://download.pytorch.org/whl/cu124"
    return "https://download.pytorch.org/whl/cu118"


def setup(python_exe: str, ext_dir: str, gpu_sm: int) -> None:
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"
    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    venv_python = python_exe_in_venv(venv)

    # Upgrade pip/setuptools/wheel inside venv using venv python -m pip
    print("[setup] Upgrading pip, setuptools, wheel in venv...")
    try:
        pip_with_venv_python(venv_python, "install", "--upgrade", "pip", "setuptools", "wheel")
    except Exception as exc:
        print("[setup] Failed to upgrade pip/setuptools/wheel inside venv.")
        print("[setup] Please run this command manually and paste the output:")
        print(f"{venv_python} -m pip install --upgrade pip setuptools wheel")
        raise

    # Install PyTorch (index chosen by GPU SM)
    idx = torch_index(gpu_sm)
    print("[setup] torch index:", idx)
    try:
        pip_with_venv_python(venv_python, "install", "torch", "torchvision", "--index-url", idx)
    except Exception as exc:
        print("[setup] PyTorch install failed; you can try installing manually with the printed index.")
        raise

    # Core deps
    core_deps = [
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
    ]
    try:
        pip_with_venv_python(venv_python, "install", *core_deps)
    except Exception as exc:
        print("[setup] Core dependency installation failed.")
        raise

    # Optional xformers
    try:
        pip_with_venv_python(venv_python, "install", "xformers")
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
