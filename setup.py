# setup.py
import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

EXT_ID = "hunyuan_t2i_turbo_modly"
HF_REPO = "TencentARC/HunyuanDiT-Turbo"
DOWNLOAD_CHECK = "config.json"   # sentinel file inside HF repo to verify download
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

def _pip_install(packages):
    """
    Install packages using the same Python interpreter.
    Uses --upgrade to ensure latest compatible versions.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    log("Installing Python packages: " + " ".join(packages))
    subprocess.check_call(cmd)

def _deps_installed() -> bool:
    return os.path.exists(_marker_path())

def _write_marker():
    try:
        with open(_marker_path(), "w", encoding="utf-8") as f:
            json.dump({"installed_by": EXT_ID, "python": sys.executable}, f)
    except Exception as e:
        log(f"Failed to write deps marker: {e}")
        raise

def _verify_downloaded_model(path: str) -> bool:
    # Check for sentinel file anywhere under path
    for root, _, files in os.walk(path):
        if DOWNLOAD_CHECK in files:
            return True
    return False

def install():
    """
    Full install: installs Python deps (if needed) and downloads model weights into
    extension/models/<sanitized_repo>. This function is idempotent.
    """
    root = _extension_root()
    final_dir = _final_model_dir()
    marker = _marker_path()

    # 1) Install dependencies if not already installed
    if not _deps_installed():
        log("Dependencies not found. Installing required Python packages now.")
        try:
            # Adjust packages list as needed for your environment.
            # Remove 'torch' if Modly provides a specific torch wheel or you want users to preinstall.
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

        _write_marker()
        log("Dependencies installed successfully.")

    else:
        log("Dependencies already installed; skipping pip install.")

    # 2) Download model weights if not present
    if os.path.exists(final_dir) and _verify_downloaded_model(final_dir):
        log(f"Model already present at {final_dir}; skipping download.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="hf_download_")
    try:
        log(f"Downloading Hugging Face repo {HF_REPO} into temporary directory...")
        # Use snapshot_download for robust download; this is invoked via python -m huggingface_hub.snapshot_download
        subprocess.check_call([
            sys.executable, "-m", "huggingface_hub.snapshot_download",
            "--repo_id", HF_REPO,
            "--local_dir", tmp_dir,
            "--local_dir_use_symlinks", "False"
        ])

        # Verify sentinel file exists somewhere under tmp_dir
        if not _verify_downloaded_model(tmp_dir):
            log(f"Warning: {DOWNLOAD_CHECK} not found in downloaded repo. Proceeding but verify contents.")

        # Remove existing final_dir if present, then move tmp_dir into place
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
# setup.py
import os
import sys
import shutil
import tempfile
import subprocess

EXT_ID = "hunyuan_t2i_turbo_modly"
HF_REPO = "TencentARC/HunyuanDiT-Turbo"
# file to check inside the downloaded repo to consider it complete
DOWNLOAD_CHECK = "config.json"

def _sanitize_repo_name(repo_id: str) -> str:
    # replace slashes with double underscore to avoid collisions
    return repo_id.replace("/", "__")

def log(msg: str):
    print(f"[{EXT_ID} setup] {msg}")

def install():
    """
    Called by Modly when the user clicks Download/Install.
    Downloads the HF repo into: <extension_root>/models/<sanitized_repo>
    """
    log("Starting install")

    # 1) Ensure huggingface_hub and required libs are available
    log("Ensuring required packages are installed...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "huggingface_hub", "diffusers>=0.27.0", "transformers>=4.39.0", "safetensors"
    ])

    # 2) Prepare paths
    root = os.path.dirname(__file__)
    models_root = os.path.join(root, "models")
    os.makedirs(models_root, exist_ok=True)

    target_name = _sanitize_repo_name(HF_REPO)
    final_dir = os.path.join(models_root, target_name)

    # If download_check exists, skip download
    check_path = os.path.join(final_dir, DOWNLOAD_CHECK)
    if os.path.exists(check_path):
        log(f"Model already present and valid at {final_dir}; skipping download.")
        return

    # 3) Download to a temporary directory first (atomic move)
    tmp_dir = tempfile.mkdtemp(prefix="hf_download_")
    try:
        log(f"Downloading {HF_REPO} into temporary dir {tmp_dir} ...")
        # Use huggingface_hub snapshot_download for robust repo download
        subprocess.check_call([
            sys.executable, "-m", "huggingface_hub.snapshot_download",
            "--repo_id", HF_REPO,
            "--cache_dir", tmp_dir,
            "--local_dir", tmp_dir,
            "--local_dir_use_symlinks", "False"
        ])

        # After download, verify the expected file exists
        expected = os.path.join(tmp_dir, DOWNLOAD_CHECK)
        if not os.path.exists(expected):
            # Some repos put files in subfolders; try to find the file anywhere under tmp_dir
            found = False
            for root_dir, _, files in os.walk(tmp_dir):
                if DOWNLOAD_CHECK in files:
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Download completed but {DOWNLOAD_CHECK} not found in {tmp_dir}")

        # 4) Move tmp_dir contents to final_dir (remove old if present)
        if os.path.exists(final_dir):
            log(f"Removing existing model dir at {final_dir}")
            shutil.rmtree(final_dir, ignore_errors=True)

        log(f"Moving downloaded model to {final_dir}")
        shutil.move(tmp_dir, final_dir)
        log("Download and install complete")

    except Exception as e:
        log(f"Error during download: {e}")
        # cleanup tmp_dir if it still exists
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        # tmp_dir moved on success; ensure no leftover
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def uninstall():
    """
    Remove the model folder from this extension.
    Called by Modly when uninstalling the extension.
    """
    log("Uninstalling extension and removing model files...")
    root = os.path.dirname(__file__)
    models_root = os.path.join(root, "models")
    target_name = _sanitize_repo_name(HF_REPO)
    final_dir = os.path.join(models_root, target_name)

    if os.path.exists(final_dir):
        log(f"Removing {final_dir}")
        shutil.rmtree(final_dir, ignore_errors=True)
    else:
        log("No model directory found to remove.")

    log("Uninstall complete")
