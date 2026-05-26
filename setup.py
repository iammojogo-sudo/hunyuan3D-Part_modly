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
