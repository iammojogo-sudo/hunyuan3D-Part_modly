#!/usr/bin/env python3
"""
Automated install/uninstall entrypoint for HunyuanDiT‑Turbo Modly extension.

Behavior:
- Fully automated: Modly calls this during Install; no manual steps required.
- Creates a venv inside the extension folder and uses the venv python for all installs and downloads.
- Upgrades pip/setuptools/wheel inside the venv.
- Installs huggingface_hub into the venv BEFORE calling snapshot_download.
- Calls snapshot_download by importing and invoking the function via `python -c` (avoids `-m` entrypoint issues).
- Retries network operations with backoff.
- Writes a small marker file for idempotency and prints clear logs for Modly UI.

Supported invocations:
  python setup.py '{"python_exe":"<path>","ext_dir":"<path>","gpu_sm":89}'
  python setup.py <python_exe> <ext_dir> <gpu_sm>
  python setup.py install
  python setup.py uninstall
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ---------- Configuration ----------
EXT_ID = "hunyuan_t2i_turbo_modly"
HF_REPO = "TencentARC/HunyuanDiT-Turbo"
DOWNLOAD_CHECK = "config.json"
DEPS_MARKER = ".deps_installed_v1"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 4.0  # seconds base

IS_WIN = sys.platform.startswith("win")

# ---------- Logging ----------
def log(msg: str) -> None:
    print(f"[{EXT_ID}] {msg}", flush=True)

# ---------- Path helpers ----------
def root_path(ext_dir: Optional[str]) -> Path:
    return Path(ext_dir).resolve() if ext_dir else Path(__file__).parent.resolve()

def venv_dir(root: Path) -> Path:
    return root / "venv"

def venv_python(venv: Path) -> str:
    return str(venv / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python"))

def venv_pip(venv: Path) -> str:
    return str(venv / ("Scripts" if IS_WIN else "bin") / ("pip.exe" if IS_WIN else "pip"))

# ---------- Subprocess helpers ----------
def run(cmd, check=True, capture=False, env=None, cwd=None):
    cmd_list = list(map(str, cmd))
    log("Running: " + " ".join(cmd_list))
    return subprocess.run(cmd_list, check=check, capture_output=capture, text=True, env=env, cwd=cwd)

def run_with_retry(cmd, attempts=RETRY_ATTEMPTS, backoff=RETRY_BACKOFF, **kwargs):
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return run(cmd, **kwargs)
        except subprocess.CalledProcessError as e:
            last_exc = e
            log(f"Command failed (attempt {i}/{attempts}): {e}")
            if i < attempts:
                sleep = backoff * i
                log(f"Retrying in {sleep:.0f}s...")
                time.sleep(sleep)
    raise last_exc

# ---------- Verification helpers ----------
def verify_model_downloaded(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.name == DOWNLOAD_CHECK:
            return True
    return False

# ---------- Core steps ----------
def create_venv(python_exe: str, venv_path: Path) -> None:
    if venv_path.exists():
        log(f"Venv already exists at {venv_path}; skipping creation.")
        return
    log(f"Creating venv at {venv_path} using {python_exe}")
    run_with_retry([python_exe, "-m", "venv", str(venv_path)])

def upgrade_pip_and_tools(vp: str) -> None:
    try:
        run_with_retry([vp, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    except subprocess.CalledProcessError:
        log("Warning: failed to upgrade pip/setuptools/wheel; continuing.")

def pip_install(venv: Path, packages: list[str]) -> None:
    pip = venv_pip(venv)
    cmd = [pip, "install", "--upgrade"] + packages
    run_with_retry(cmd)

def ensure_hf_and_core(venv: Path) -> None:
    # Install huggingface_hub first (critical)
    log("Installing huggingface_hub into venv (required for snapshot_download)...")
    pip_install(venv, ["huggingface_hub>=0.16.4"])
    # Install core runtime deps
    core = [
        "diffusers>=0.27.0",
        "transformers>=4.39.0",
        "accelerate",
        "safetensors",
        "Pillow",
        "numpy",
        "tqdm",
    ]
    log("Installing core runtime dependencies into venv...")
    pip_install(venv, core)

    # Verification: pip show and import check
    try:
        out = run([venv_pip(venv), "show", "huggingface_hub"], capture=True)
        log("huggingface_hub installed:\n" + (out.stdout or out.stderr))
    except Exception:
        log("Warning: pip show huggingface_hub failed.")

    try:
        out = run([venv_python(venv), "-c", "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"], capture=True)
        log("huggingface_hub import check:\n" + (out.stdout or out.stderr))
    except Exception as e:
        log(f"ERROR: huggingface_hub import check failed: {e}")
        raise

def download_hf_repo_using_function(venv_python: str, hf_repo: str, dest_parent: Path, download_check: str, attempts=RETRY_ATTEMPTS, backoff=RETRY_BACKOFF) -> Path:
    """
    Use the venv python to import and call snapshot_download directly via -c.
    This avoids `python -m huggingface_hub.snapshot_download` which is not a runnable module on some installs.
    Returns the final model directory path.
    """
    dest_parent.mkdir(parents=True, exist_ok=True)
    final_dir = dest_parent / hf_repo.replace("/", "__")
    if final_dir.exists() and any(p.name == download_check for p in final_dir.rglob("*") if p.is_file()):
        log(f"Model already present at {final_dir}; skipping download.")
        return final_dir

    tmp = Path(tempfile.mkdtemp(prefix=f"{EXT_ID}_hf_"))
    try:
        # Build inline python command that calls snapshot_download
        cmd_template = (
            "from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id={repo!r}, local_dir={local_dir!r}, local_dir_use_symlinks=False)"
        ).format(repo=hf_repo, local_dir=str(tmp))

        last_exc = None
        for i in range(1, attempts + 1):
            try:
                log(f"Downloading HF repo {hf_repo} into temporary dir {tmp} (attempt {i}/{attempts})...")
                run([venv_python, "-c", cmd_template], check=True)
                # verify presence of download_check
                if not any(p.name == download_check for p in tmp.rglob("*") if p.is_file()):
                    log(f"Warning: {download_check} not found in downloaded repo (attempt {i}).")
                # move into place
                if final_dir.exists():
                    shutil.rmtree(final_dir, ignore_errors=True)
                shutil.move(str(tmp), str(final_dir))
                log(f"Model downloaded and moved into place: {final_dir}")
                return final_dir
            except subprocess.CalledProcessError as e:
                last_exc = e
                log(f"snapshot_download failed (attempt {i}/{attempts}): {e}")
                if i < attempts:
                    sleep = backoff * i
                    log(f"Retrying in {sleep:.0f}s...")
                    time.sleep(sleep)
        # all attempts failed
        raise last_exc
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

# ---------- Public install/uninstall ----------
def install(python_exe: str, ext_dir: str, gpu_sm: int) -> None:
    root = root_path(ext_dir)
    venv = venv_dir(root)
    marker = root / DEPS_MARKER
    models_parent = root / "models"

    create_venv(python_exe, venv)
    vp = venv_python(venv)

    upgrade_pip_and_tools(vp)
    ensure_hf_and_core(venv)

    # Download HF repo into models/<sanitized_repo>
    try:
        download_hf_repo_using_function(vp, HF_REPO, models_parent, DOWNLOAD_CHECK)
    except subprocess.CalledProcessError as e:
        log(f"Error during snapshot_download: {e}")
        raise

    # Write marker for idempotency
    try:
        if not marker.exists():
            marker.write_text(json.dumps({"installed_by": EXT_ID, "python": vp}), encoding="utf-8")
            log(f"Created deps marker at {marker}")
        else:
            log("Deps marker already present.")
    except Exception as e:
        log(f"Warning: could not write deps marker: {e}")

    log("Install complete.")

def uninstall(ext_dir: str) -> None:
    root = root_path(ext_dir)
    venv = venv_dir(root)
    model_dir = root / "models" / EXT_ID
    marker = root / DEPS_MARKER

    if model_dir.exists():
        log(f"Removing model directory {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        log("No model directory found to remove.")

    if venv.exists():
        log(f"Removing venv directory {venv}")
        shutil.rmtree(venv, ignore_errors=True)
    else:
        log("No venv directory found to remove.")

    if marker.exists():
        try:
            marker.unlink()
            log(f"Removed dependency marker {marker}")
        except Exception as e:
            log(f"Failed to remove marker: {e}")

    log("Uninstall complete.")

# ---------- CLI entry ----------
def main() -> None:
    try:
        # JSON arg form
        if len(sys.argv) == 2 and sys.argv[1].startswith("{"):
            try:
                args = json.loads(sys.argv[1])
            except Exception as exc:
                log(f"Invalid JSON args: {exc}")
                sys.exit(2)
            python_exe = args.get("python_exe", sys.executable)
            ext_dir = args.get("ext_dir", str(Path(__file__).parent))
            gpu_sm = int(args.get("gpu_sm", 70))
            install(python_exe, ext_dir, gpu_sm)
            return

        # Positional form: python setup.py <python_exe> <ext_dir> <gpu_sm>
        if len(sys.argv) >= 4:
            python_exe = sys.argv[1]
            ext_dir = sys.argv[2]
            try:
                gpu_sm = int(sys.argv[3])
            except Exception:
                gpu_sm = 70
            install(python_exe, ext_dir, gpu_sm)
            return

        # Simple commands
        if len(sys.argv) == 2:
            cmd = sys.argv[1].lower()
            if cmd == "install":
                install(sys.executable, str(Path(__file__).parent), 70)
                return
            if cmd == "uninstall":
                uninstall(str(Path(__file__).parent))
                return

        print("Usage:")
        print("  python setup.py '{\"python_exe\":\"...\",\"ext_dir\":\"...\",\"gpu_sm\":89}'")
        print("  python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print("  python setup.py install")
        print("  python setup.py uninstall")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        # Clear, non-ambiguous exit for Modly to show failure reason
        log(f"Setup failed: {e}")
        sys.exit(2)
    except Exception as e:
        log(f"Unexpected error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
