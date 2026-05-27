# hunyuan_t2i_turbo_modly

Modly extension for text-to-image generation using HunyuanDiT v1.2 Turbo (distilled).

## Usage

1. Install the extension from the Models page in Modly
2. Download the model weights (about 15GB, downloads automatically on first run)
3. In Workflows, add the **Generate Image** node
4. Connect any image node as the input — the image isn't used, Modly just needs something there
5. Type your prompt in the Prompt field and hit Run
6. Output PNG lands in your workspace folder

## Notes

- Needs about 6GB VRAM minimum
- First generation will be slow while the model loads
- Seed -1 means random, set a number to get reproducible results
- The input image is ignored, this is text-to-image only

## Model

Weights from [Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled) on HuggingFace.
