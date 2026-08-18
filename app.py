#!/usr/bin/env python
"""Gradio front end for all three models.

    python app.py

Two tabs.  **Generate** runs one model.  **Compare** runs the same prompt and
the same seed through every available model, which is the whole point of the
project: with the noise, the text encoder, the VAE and the sampler all shared,
what is left between the columns is the denoiser itself.

Only one denoiser is kept in VRAM at a time.  Stable Diffusion's UNet is 1.7 GB
in fp16 and there are 8 GB total, so holding all three plus the frozen encoders
would not fit -- switching evicts the previous model.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import gradio as gr
import torch

from ttimg.models.encoders import TextConditioner
from ttimg.models.registry import MODEL_LABELS, build_pipeline, find_latest_checkpoint

DEFAULT_DIT_RUN = Path("checkpoints/dit_small")
DEFAULT_LORA = Path("checkpoints/sd15_lora_cub/lora.safetensors")

EXAMPLE_PROMPTS = [
    "this bird has a bright red belly with black wings and a short pointed beak",
    "a small yellow bird with a grey head and a thin sharp bill",
    "a large white seabird with long narrow wings and a hooked orange beak",
    "this small bird has an iridescent green throat and a very long thin bill",
    "a brown speckled bird with a white throat and striped wingbars",
]


class ModelManager:
    """Builds pipelines on demand, holding at most one denoiser in VRAM.

    The text encoder and VAE are created once and handed to every pipeline, so
    switching models moves ~1.7 GB rather than reloading ~2.5 GB of frozen
    weights each time.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.conditioner = TextConditioner(device=device, dtype=dtype)
        self._current: str | None = None
        self._pipeline = None
        self.dit_checkpoint = find_latest_checkpoint(DEFAULT_DIT_RUN) if DEFAULT_DIT_RUN.exists() else None
        self.lora_path = DEFAULT_LORA if DEFAULT_LORA.exists() else None

    def available(self) -> list[str]:
        kinds = ["sd15"]
        if self.lora_path:
            kinds.append("sd15_lora")
        if self.dit_checkpoint:
            kinds.append("dit")
        return kinds

    def get(self, kind: str):
        if kind == self._current and self._pipeline is not None:
            return self._pipeline

        # Evict before loading -- the card cannot hold both.
        self._pipeline = None
        self._current = None
        gc.collect()
        torch.cuda.empty_cache()

        pipeline = build_pipeline(
            kind=kind,
            checkpoint=self.dit_checkpoint,
            lora_path=self.lora_path,
            device=self.device,
            dtype=self.dtype,
            conditioner=self.conditioner,
        )
        self._pipeline, self._current = pipeline, kind
        return pipeline


def build_ui(manager: ModelManager) -> gr.Blocks:
    kinds = manager.available()
    choices = [(MODEL_LABELS[k], k) for k in kinds]

    missing = []
    if "sd15_lora" not in kinds:
        missing.append("LoRA not found - run `python scripts/train.py --config-name train_lora`")
    if "dit" not in kinds:
        missing.append("DiT checkpoint not found - run `python scripts/train.py --config-name train_dit`")

    def generate(prompt, kind, steps, guidance, seed, count, randomize):
        if not prompt.strip():
            return [], "Enter a caption first."
        actual_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if randomize else int(seed)
        pipeline = manager.get(kind)
        images = pipeline.generate(
            [prompt] * int(count), steps=int(steps), guidance_scale=float(guidance), seed=actual_seed
        )
        return images, f"{MODEL_LABELS[kind]} | {int(steps)} steps | CFG {guidance} | seed {actual_seed}"

    def compare(prompt, steps, guidance, seed, randomize):
        if not prompt.strip():
            return [], "Enter a caption first."
        actual_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if randomize else int(seed)
        gallery = []
        for kind in kinds:
            pipeline = manager.get(kind)
            image = pipeline.generate(
                [prompt], steps=int(steps), guidance_scale=float(guidance), seed=actual_seed
            )[0]
            gallery.append((image, MODEL_LABELS[kind]))
        return gallery, f"Same prompt, same seed ({actual_seed}) through {len(kinds)} model(s)."

    with gr.Blocks(title="CUB Text-to-Image") as demo:
        gr.Markdown(
            "# Birds from captions\n"
            "Two models over one shared latent space: Stable Diffusion 1.5 fine-tuned with "
            "LoRA, and a diffusion transformer trained from scratch on CUB-200."
        )
        if missing:
            gr.Markdown("> **Not yet trained:** " + " · ".join(missing))

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Caption",
                        value=EXAMPLE_PROMPTS[0],
                        lines=2,
                        placeholder="describe a bird...",
                    )
                    gr.Examples(EXAMPLE_PROMPTS, inputs=prompt, label="Try one")
                with gr.Column(scale=2):
                    model = gr.Dropdown(choices, value=kinds[0], label="Model")
                    count = gr.Slider(1, 8, value=4, step=1, label="Images")

            with gr.Row():
                steps = gr.Slider(4, 100, value=30, step=1, label="Steps")
                guidance = gr.Slider(1.0, 15.0, value=4.0, step=0.5, label="Guidance (CFG)")
                seed = gr.Number(value=1234, precision=0, label="Seed")
                randomize = gr.Checkbox(value=False, label="Random seed")

            run = gr.Button("Generate", variant="primary")
            gallery = gr.Gallery(label="Results", columns=4, height=430)
            status = gr.Markdown()

            run.click(
                generate,
                [prompt, model, steps, guidance, seed, count, randomize],
                [gallery, status],
            )

        with gr.Tab("Compare"):
            gr.Markdown(
                "Same caption, same starting noise, same sampler, through every model. "
                "Differences here are differences in the denoiser."
            )
            compare_prompt = gr.Textbox(label="Caption", value=EXAMPLE_PROMPTS[0], lines=2)
            with gr.Row():
                compare_steps = gr.Slider(4, 100, value=30, step=1, label="Steps")
                compare_guidance = gr.Slider(1.0, 15.0, value=4.0, step=0.5, label="Guidance (CFG)")
                compare_seed = gr.Number(value=1234, precision=0, label="Seed")
                compare_random = gr.Checkbox(value=False, label="Random seed")

            compare_run = gr.Button("Compare models", variant="primary")
            compare_gallery = gr.Gallery(label="Side by side", columns=len(kinds) or 1, height=430)
            compare_status = gr.Markdown()

            compare_run.click(
                compare,
                [compare_prompt, compare_steps, compare_guidance, compare_seed, compare_random],
                [compare_gallery, compare_status],
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    args = parser.parse_args()

    manager = ModelManager(device=args.device)
    print(f"[app] available models: {', '.join(manager.available())}")
    build_ui(manager).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
