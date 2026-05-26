# setup.py
"""
Install/uninstall entrypoint used by Modly.
- install() will install Python deps (idempotent marker) and snapshot_download
  the HF repo into: <extension_root>/models/<manifest_id>/
- uninstall() will remove that model folder and the deps marker.
"""
import json
import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

# Configuration (keep in sync with manifest.json id)
EXT_ID = "hunyuan_t2i_turbo_modly"
HF_REPO = "TencentARC/HunyuanDiT-Turbo"
DOWNLOAD_CHECK = "config.json"
DEPS_MARKER = ".deps_installed_v1"

def _root() -> Path:
    return Path(__file__).parent.resolve()

def _models_dir() -> Path:
    d = _root() / "models" / EXT_ID
    d.mkdir(parents=True, exist_ok=True)
    return d

def _marker_path() -> Path:
    return _root() / DEPS_MARKER

def _verify_downloaded_model(path: Path) -> bool:
    # Look for the download_check file anywhere under the model dir
    for p in path.rglob("*"):
        if p.is_file() and p.name == DOWNLOAD_CHECK:
            return True
    return False

def _pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    subprocess.check_call(cmd)

def install():
    root = _root()
    model_dir = _models_dir()
    marker = _marker_path()

    # 1) Install deps once
    if not marker.exists():
        print(f"[{EXT_ID}] Installing Python dependencies...", flush=True)
        try:
            _pip_install([
                "huggingface_hub>=0.16.4",
                "diffusers>=0.27.0",
                "transformers>=4.39.0",
                "accelerate",
                "safetensors"
            ])
        except subprocess.CalledProcessError as e:
            print(f"[{EXT_ID}] pip install failed: {e}", flush=True)
            raise
        # create marker
        marker.write_text(json.dumps({"installed_by": EXT_ID, "python": sys.executable}), encoding="utf-8")
        print(f"[{EXT_ID}] Dependencies installed; marker created.", flush=True)
    else:
        print(f"[{EXT_ID}] Dependencies marker found; skipping pip install.", flush=True)

    # 2) Download HF repo into models/<ext_id> (idempotent)
    if model_dir.exists() and _verify_downloaded_model(model_dir):
        print(f"[{EXT_ID}] Model already present at {model_dir}; skipping download.", flush=True)
        return

    tmp = Path(tempfile.mkdtemp(prefix=f"{EXT_ID}_hf_"))
    try:
        print(f"[{EXT_ID}] Downloading {HF_REPO} into temporary dir {tmp} ...", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "huggingface_hub.snapshot_download",
            "--repo_id", HF_REPO,
            "--local_dir", str(tmp),
            "--local_dir_use_symlinks", "False"
        ])
        if not _verify_downloaded_model(tmp):
            print(f"[{EXT_ID}] Warning: {DOWNLOAD_CHECK} not found in downloaded repo. Proceeding.", flush=True)

        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        shutil.move(str(tmp), str(model_dir))
        print(f"[{EXT_ID}] Model moved into place: {model_dir}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[{EXT_ID}] Error during snapshot_download: {e}", flush=True)
        raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

def uninstall():
    model_dir = _root() / "models" / EXT_ID
    marker = _marker_path()

    if model_dir.exists():
        print(f"[{EXT_ID}] Removing model directory {model_dir}", flush=True)
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print(f"[{EXT_ID}] No model directory to remove.", flush=True)

    if marker.exists():
        try:
            marker.unlink()
            print(f"[{EXT_ID}] Removed deps marker {marker}", flush=True)
        except Exception as e:
            print(f"[{EXT_ID}] Failed to remove marker: {e}", flush=True)

if __name__ == "__main__":
    # Allow Modly to call: python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":89}'
    # or: python setup.py install
    if len(sys.argv) == 2:
        cmd = sys.argv[1].lower()
        if cmd == "install":
            install()
        elif cmd == "uninstall":
            uninstall()
        else:
            print("Usage: python setup.py [install|uninstall]")
    elif len(sys.argv) == 4:
        # positional args: python_exe ext_dir gpu_sm  (Modly sometimes calls like this)
        # We ignore python_exe/gpu_sm here because we run in the extension's Python.
        install()
    elif len(sys.argv) == 2 and sys.argv[1].startswith("{"):
        # JSON arg form
        try:
            args = json.loads(sys.argv[1])
            install()
        except Exception:
            print("Invalid JSON args")
    else:
        print("Usage: python setup.py [install|uninstall]")
