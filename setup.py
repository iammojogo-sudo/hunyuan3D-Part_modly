import os
import sys
import subprocess
import shutil

EXTENSION_NAME = "hunyuan_t2i_turbo"

def log(msg: str):
    print(f"[{EXTENSION_NAME} setup] {msg}")

def install():
    log("Starting installation")

    # -----------------------------
    # 1. Install Python dependencies
    # -----------------------------
    log("Installing Python dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors",
        "huggingface_hub"
    ])

    # -----------------------------
    # 2. Create model directory
    # -----------------------------
    root = os.path.dirname(__file__)
    model_dir = os.path.join(root, "models", "hunyuan_t2i_turbo")
    os.makedirs(model_dir, exist_ok=True)
    log(f"Model directory created at: {model_dir}")

    # -----------------------------
    # 3. Download model weights
    # -----------------------------
    log("Downloading HunyuanDiT-Turbo model from HuggingFace...")
    subprocess.check_call([
        sys.executable, "-m", "huggingface_hub.download",
        "--repo-id", "TencentARC/HunyuanDiT-Turbo",
        "--local-dir", model_dir,
        "--local-dir-use-symlinks", "False"
    ])

    log("Installation complete")

def uninstall():
    log("Uninstalling extension...")

    root = os.path.dirname(__file__)
    model_dir = os.path.join(root, "models", "hunyuan_t2i_turbo")

    if os.path.exists(model_dir):
        log("Removing model directory...")
        shutil.rmtree(model_dir, ignore_errors=True)

    log("Uninstall complete")
