# setup.py
import os
import sys
import json
import shutil
import tempfile
import subprocess

EXT_ID = "hunyuan_t2i_turbo_modly"
HF_REPO = "TencentARC/HunyuanDiT-Turbo"
DOWNLOAD_CHECK = "config.json"   # sentinel file to verify download
DEPS_MARKER = ".deps_installed_v1"

def log(msg: str):
    print(f"[{EXT_ID} setup] {msg}", flush=True)

def _sanitize_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")

def _extension_root() -> str:
    return os.path.dirname(__file__)

def _models_root() -> str:
    root = _extension_root()
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def _final_model_dir() -> str:
    return os.path.join(_models_root(), _sanitize_repo_name(HF_REPO))

def _marker_path() -> str:
    return os.path.join(_extension_root(), DEPS_MARKER)

def _verify_downloaded_model(path: str) -> bool:
    for root, _, files in os.walk(path):
        if DOWNLOAD_CHECK in files:
            return True
    return False

def _pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    log("Installing Python packages: " + " ".join(packages))
    subprocess.check_call(cmd)

def install():
    """
    Full install: installs Python deps and downloads model weights into
    extension/models/<sanitized_repo>. Idempotent.
    """
    root = _extension_root()
    final_dir = _final_model_dir()
    marker = _marker_path()

    # 1) Install dependencies (only if marker missing)
    if not os.path.exists(marker):
        log("Dependencies not found. Installing required Python packages now.")
        try:
            _pip_install([
                "huggingface_hub>=0.16.4",
                "diffusers>=0.27.0",
                "transformers>=4.39.0",
                "accelerate",
                "safetensors",
                "torch"
            ])
        except subprocess.CalledProcessError as e:
            log(f"pip install failed: {e}")
            raise

        try:
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"installed_by": EXT_ID, "python": sys.executable}, f)
            log(f"Dependencies installed. Marker created at {marker}")
        except Exception as e:
            log(f"Failed to write deps marker: {e}")
            raise
    else:
        log("Dependencies already installed; skipping pip install.")

    # 2) Download model weights if not present or incomplete
    if os.path.exists(final_dir) and _verify_downloaded_model(final_dir):
        log(f"Model already present at {final_dir}; skipping download.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="hf_download_")
    try:
        log(f"Downloading Hugging Face repo {HF_REPO} into temporary directory...")
        subprocess.check_call([
            sys.executable, "-m", "huggingface_hub.snapshot_download",
            "--repo_id", HF_REPO,
            "--local_dir", tmp_dir,
            "--local_dir_use_symlinks", "False"
        ])

        if not _verify_downloaded_model(tmp_dir):
            log(f"Warning: {DOWNLOAD_CHECK} not found in downloaded repo. Proceeding but verify contents.")

        if os.path.exists(final_dir):
            log(f"Removing existing model directory at {final_dir}")
            shutil.rmtree(final_dir, ignore_errors=True)

        log(f"Moving downloaded model into place at {final_dir}")
        shutil.move(tmp_dir, final_dir)
        log("Model download and installation complete.")

    except subprocess.CalledProcessError as e:
        log(f"Error during model download: {e}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        log(f"Unexpected error during download: {e}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def uninstall():
    """
    Remove model folder and dependency marker. Do not attempt to uninstall pip packages.
    """
    final_dir = _final_model_dir()
    marker = _marker_path()

    if os.path.exists(final_dir):
        log(f"Removing model directory {final_dir}")
        shutil.rmtree(final_dir, ignore_errors=True)
    else:
        log("No model directory found to remove.")

    if os.path.exists(marker):
        try:
            os.remove(marker)
            log(f"Removed dependency marker {marker}")
        except Exception as e:
            log(f"Failed to remove marker: {e}")

    log("Uninstall complete.")
