# Birds from captions

Two text-to-image models over one shared latent space, on the CUB-200-2011 bird
dataset:

1. **Stable Diffusion 1.5 fine-tuned with LoRA** — the pragmatic path.
2. **A diffusion transformer trained from scratch** — ~56M parameters, no
   pretrained denoiser weights, rectified-flow objective.

Both run behind one Gradio app that can put them side by side on the same prompt
and the same starting noise.

Built to run on a single **8 GB laptop GPU** (RTX 5050, Blackwell `sm_120`).

---

## The idea

The expensive, generic parts of a text-to-image system — the image autoencoder
and the text encoder — are *shared*, and only the denoiser differs:

```
caption → CLIP ViT-L/14 text encoder ─┐
                                      ├→ [ denoiser ] → latents → SD VAE → image
noise ───────────────────────────────┘        ↑
                                     ┌────────┴────────┐
                                SD 1.5 UNet        Scratch DiT
                                  (+ LoRA)        (flow matching)
```

Everything else is common: the precomputed latent cache, the text embeddings,
the classifier-free guidance loop, the decoder, the evaluation harness and the
GUI. The from-scratch model is a *second module behind one interface*
(`ttimg/models/interface.py`), not a second project.

This is also what makes the comparison honest. When the GUI renders both models
on one prompt, the noise, text encoder, sampler and decoder are identical — the
only thing that differs is the denoiser.

## Why these choices

| Decision | Reasoning |
|---|---|
| **SD 1.5**, not SDXL | 2 GB in fp16 vs 7 GB. Fits 8 GB at 512px with room to train. |
| **Precompute latents, encode text live** | Latents: expensive to compute, cheap to store (950 MB). Text: the reverse — caching 117,880 captions at 77×768 would be **14 GB**, while a CLIP text pass is ~2 ms. So the VAE never occupies VRAM during training. |
| **Rectified flow** for the scratch model | Straight paths between noise and data mean Euler integration is *exact* (verified in `tests/`), no beta schedules or posterior variances to get wrong, and good samples at 20–30 steps. |
| **Cross-attention**, not pooled conditioning | Pooling CLIP's 77 tokens into one vector destroys exactly the per-attribute detail that makes CUB interesting. "Red belly, black wings" must be able to place the red on the belly. |
| **adaLN-single** (PixArt) | Standard DiT spends `6·d²` params per block on timestep modulation — at d=512, depth 12, that is 19M params, a third of the model. One shared MLP plus small per-block offsets buys that back as capacity. |
| **Bounding-box crop** | CUB birds are often small in cluttered frames. Cropping so the bird fills ~75% of the frame (StackGAN convention) stops a small model wasting capacity on branches and sky. |
| **QK-norm, EMA, logit-normal timesteps** | Cheap, standard, and each worth more than they cost: bounded attention logits in bf16, cleaner samples, and training effort spent mid-trajectory where the task is hard. |

## Setup

Requires PyTorch with Blackwell (`sm_120`) support:

```bash
pip install diffusers transformers accelerate peft gradio tensorboard
```

> On a shared venv holding a torch nightly, run `pip install --dry-run ...` first
> and confirm `torch` is **not** in the install list — a stable wheel without
> `sm_120` will silently break the GPU.

## Usage

```bash
# 1. Download CUB and build the latent caches (~35 min, one time)
python scripts/prepare_data.py

# 2a. Fine-tune SD 1.5 with LoRA
python scripts/train.py --config-name train_lora

# 2b. Train the DiT from scratch
python scripts/train.py --config-name train_dit

#     ...but first, the correctness gate: memorise 10 images.
#     If this does not converge, there is a bug. Find it here, not after 6 hours.
python scripts/train.py --config-name train_dit overfit=10 run_name=overfit_check

# 3. Watch training (all local — no wandb)
tensorboard --logdir logs/
python scripts/plot_curves.py logs/dit_small     # static PNGs for the README

# 4. Generate
python scripts/sample.py model_kind=sd15 prompt="a small blue bird with a white belly"
python scripts/sample.py model_kind=dit checkpoint=checkpoints/dit_small/last.ckpt
python scripts/sample.py mode=sweep_attribute    # does one changed word change the image?
python scripts/sample.py mode=sweep_guidance     # find the usable CFG range (3.0 here)

# 5. GUI
python app.py
```

Everything is Hydra-configured; any value can be overridden on the command line:

```bash
python scripts/train.py --config-name train_dit model=dit_base data.batch_size=16
```

## Layout

```
configs/          Hydra configs — model, data, diffusion, logger, per-task
ttimg/
  data/           CUB download & index, bbox crop, latent precompute, dataset
  models/         interface.py (the seam), dit.py (from scratch), sd15.py (LoRA),
                  encoders.py (shared VAE + CLIP), registry.py (load by name)
  diffusion/      flow.py (rectified flow), sampler.py (Euler + DDIM, CFG)
  training/       Lightning modules, EMA, sample-grid callback
  eval/           CLIP score
  pipeline.py     caption → image, model-agnostic
scripts/          prepare_data.py, train.py, sample.py, plot_curves.py
tests/            oracle tests for the samplers
app.py            Gradio: generate + side-by-side compare
```

## Verification

The project is built around gates that fail loudly and early:

| Gate | Check | Status |
|---|---|---|
| Environment | Blackwell `sm_120` survives the installs | ✅ |
| Latent cache | Decoded latents match originals | ✅ **27.15 dB** |
| Flip bookkeeping | Cache index 1 is the mirror of index 0 | ✅ **27.13 dB** |
| Flow sampler | Oracle denoiser recovers data exactly | ✅ **4e-8, at 1 step** |
| DDIM sampler | Oracle denoiser recovers data exactly | ✅ **2.4e-8** |
| CFG | Scale 1.0 is bit-identical to unguided | ✅ |
| DiT correctness | Memorises 10 images (10 distinct species) | ✅ **loss 1.98 → 0.0034** |
| DiT full run | 300 epochs to completion | ✅ **val 1.505 → 0.730** |

The sharpest end-to-end test is the **attribute sweep**: two captions differing
in exactly one word, rendered from the same seed. A model with working
cross-attention changes that attribute and leaves the rest of the bird alone.

