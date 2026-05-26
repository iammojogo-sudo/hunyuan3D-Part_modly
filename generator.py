import os, sys, time, uuid
from pathlib import Path
from PIL import Image
from services.generators.base import BaseGenerator, smooth_progress

_HF_REPO = "Tencent-Hunyuan/HunyuanDiT"  # upstream family; snapshot_download will pick correct folder

class HunyuanT2IGenerator(BaseGenerator):
    MODEL_ID = "hunyuan_t2i"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB = 6
    MODEL_VARIANT = "t2i-turbo"

    def is_downloaded(self):
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return (self.model_dir / "model_index.json").exists()

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        repo_id = self.hf_repo or _HF_REPO
        print("[HunyuanT2I] Downloading model snapshot:", repo_id)
        snapshot_download(repo_id=repo_id, local_dir=str(self.model_dir))
        print("[HunyuanT2I] Download complete.")

    def load(self):
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        from diffusers import DiffusionPipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        # load pipeline from local files
        self._pipe = DiffusionPipeline.from_pretrained(str(self.model_dir), local_files_only=True, torch_dtype=dtype)
        self._pipe.to(device)
        self._model = True
        print("[HunyuanT2I] Ready on", device)

    def generate(self, prompt_bytes, params, progress_cb=None, cancel_event=None):
        # prompt_bytes expected to be UTF-8 text
        prompt = prompt_bytes.decode("utf-8") if isinstance(prompt_bytes, (bytes, bytearray)) else str(prompt_bytes)
        steps = int(params.get("num_inference_steps", 20))
        seed = int(params.get("seed", 42))
        import torch
        generator = torch.Generator(device=self._pipe.device).manual_seed(seed)
        out_dir = self.outputs_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8]))
        with torch.no_grad():
            image = self._pipe(prompt, num_inference_steps=steps, generator=generator).images[0]
            image.save(str(out_path))
        return str(out_path)
