import json
import os
import sys
import time
import uuid
from pathlib import Path

# venv injection
_here = Path(__file__).parent
for _v in (_here / "venv", _here / ".venv"):
    if _v.exists():
        _sp = _v / "Lib" / "site-packages"
        if _sp.exists() and str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))
        _lib = _v / "lib"
        if _lib.exists():
            for _d in _lib.glob("python3.*"):
                _s = _d / "site-packages"
                if _s.exists() and str(_s) not in sys.path:
                    sys.path.insert(0, str(_s))
        break


def send(msg):
    print(json.dumps(msg), flush=True)

def progress(pct, step=""):
    send({"type": "progress", "pct": pct, "step": step})

def log(msg):
    send({"type": "log", "message": str(msg)})


def main():
    params_json   = sys.argv[1] if len(sys.argv) > 1 else "{}"
    models_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".modly" / "models"
    workspace_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.home() / ".modly" / "workspace"

    params = json.loads(params_json)
    prompt = str(params.get("prompt", "a beautiful landscape")).strip() or "a beautiful landscape"
    steps  = int(params.get("num_inference_steps", 20))
    seed   = int(params.get("seed", -1))
    height = int(params.get("height", 1024))
    width  = int(params.get("width", 1024))

    # weights land at MODELS_DIR/hunyuan_t2i_turbo/generate/ from the hf-download endpoint
    model_dir = models_dir / "hunyuan_t2i_turbo" / "generate"

    log("prompt='%s' steps=%d seed=%d %dx%d" % (prompt, steps, seed, width, height))

    if not (model_dir / "model_index.json").exists():
        send({"type": "error", "message": "Weights not found at %s. Download from the Models page." % model_dir})
        return

    progress(5, "Loading model...")
    import torch
    from diffusers import HunyuanDiTPipeline

    pipe = HunyuanDiTPipeline.from_pretrained(str(model_dir), torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    progress(20, "Generating...")
    gen = None
    if seed >= 0:
        gen = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=gen,
    )

    progress(90, "Saving...")
    out_dir = workspace_dir / "Workflows"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "{}_{}.png".format(int(time.time()), uuid.uuid4().hex[:8])
    out_path = out_dir / fname
    result.images[0].save(str(out_path), format="PNG")

    send({"type": "done", "output_path": str(out_path)})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        send({"type": "error", "message": str(e), "traceback": traceback.format_exc()})
        sys.exit(1)
