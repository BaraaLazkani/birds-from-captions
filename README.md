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

## Results

### From-scratch DiT

Trained from random initialisation on one 8 GB laptop GPU.

| | |
|---|---|
| Parameters | 55.6M |
| Training | 300 epochs, 104,699 steps, **9h21m** |
| Throughput | batch 32, ~122 img/s (~10.4 TFLOPS bf16 — near this card's peak) |
| Train loss | 1.6006 → **0.7188** |
| Val loss | 1.5047 → 0.7452 (best **0.7296** @ epoch 262) |

Starting loss of ~1.98 is not arbitrary: for unit-variance latents the flow
target `eps - x_0` has variance 2, so an untrained model scores 2.0 by
definition. That the run begins there is a check on the objective itself.

### Guidance is a moving target

Every progress grid during training was drawn at a fixed CFG of 4.0, and the
late-training grids looked *worse* than the mid-training ones — harsher colour,
fragmented feather texture. The loss curves showed no such degradation.

The sweep explains it. **The guidance a model wants falls as it trains.** An
under-trained model needs strong guidance to produce anything coherent; a
well-trained one over-saturates at the same scale. At CFG 1 output is muted and
vague, CFG 3 is crisp with real feather detail, and by CFG 6–8 reds posterise
into flat masses and detail dissolves.

**CFG 3.0 is the operating point.** The apparent regression was a stale sampling
setting, not a worse model — worth remembering, because a fixed preview setting
will eventually mislead you about any generative run.

### Text conditioning works

The sharpest test CUB affords: caption pairs differing in exactly one word,
rendered from identical noise. All four pairs change the intended attribute and
leave the rest of the bird alone —

| prompt A | prompt B | result |
|---|---|---|
| bright **red** belly, black wings | bright **blue** belly, black wings | ✅ |
| yellow bird, **grey** head | yellow bird, **black** head | ✅ |
| very **long thin** bill | very **short thick** bill | ✅ |
| completely **white** bird on water | completely **black** bird on water | ✅ |

This is the payoff of conditioning through cross-attention over the full 77-token
CLIP sequence rather than a pooled vector.

### SD 1.5 + LoRA

| | |
|---|---|
| Trainable parameters | **3.19M of 862.7M (0.37%)** |
| Training | 8,000 steps (~5.7 epochs), batch 8 @ 512px, ~1h50m |
| VRAM | 4.7 GB |
| Val loss | 0.1555 → **0.1500** |
| Adapter size | **6.4 MB** |

### Head to head

Same prompts, same seed, same text encoder, same VAE, same sampler — only the
denoiser differs. CLIP score is ViT-B/16 over 24 held-out validation captions.

| Model | CLIP @ CFG 3.0 | CLIP @ CFG 7.5 | Attribute pairs correct |
|---|---|---|---|
| SD 1.5 (stock) | 0.3029 | 0.3053 | — |
| **SD 1.5 + CUB LoRA** | 0.3015 | **0.3099** | ~2 / 4 |
| Scratch DiT (55.6M) | 0.2871 | 0.2920 | **4 / 4** |

**The honest summary: the LoRA wins on image quality, the from-scratch DiT wins
on caption fidelity.**

The LoRA produces sharp, photographic 512px birds and takes the best CLIP score
— unsurprising for 860M pretrained parameters adapted with 3.19M more. But it
inherits Stable Diffusion's strong prior, and that prior overrides fine caption
detail: asked for "a small yellow bird with a **black** head" it renders a yellow
bird with a yellow head.

The DiT is blurrier and works at 256px, and its CLIP score is ~6% lower. But it
has no prior to override — CUB is all it has ever seen — so it follows the
caption literally, changing exactly the attribute the prompt changed in all four
minimal pairs.

**A caveat about the metric**, since it cuts against the images: CLIP score
ranked *every* model higher at CFG 7.5 than at 3.0, including the DiT, even
though 7.5 visibly posterises the DiT's output. CLIP score rewards prompt
saturation more than human judgement does. It is a useful number, not a verdict
— which is why the sample grids stay in this README next to it.

## Dataset

[CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) (11,788
images, 200 species) with the 10-captions-per-image annotations from Reed et al.
(2016), *Learning Deep Representations of Fine-Grained Visual Descriptions*.
Research and educational use only.

## References

- Peebles & Xie, [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748) (DiT)
- Chen et al., [*PixArt-α*](https://arxiv.org/abs/2310.00426) — cross-attention conditioning, adaLN-single
- Esser et al., [*Scaling Rectified Flow Transformers*](https://arxiv.org/abs/2403.03206) (SD3) — flow matching, logit-normal timesteps
- Sehwag et al., [*Stretching Each Dollar*](https://arxiv.org/abs/2407.15811) — micro-budget diffusion training
- Xie et al., [*SANA*](https://arxiv.org/abs/2410.10629) — efficient high-resolution synthesis
- Hu et al., [*LoRA*](https://arxiv.org/abs/2106.09685)
