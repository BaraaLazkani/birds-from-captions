# Where things stand

Last updated: 2026-08-18

## TL;DR

The **from-scratch DiT finished** — 300 epochs, 9h21m, and it works.
The **LoRA fine-tune ran overnight** after it. Everything is local; no network
needed any more.

---

## Done and verified

| Gate | Result |
|---|---|
| Blackwell `sm_120` survives the pip installs | ✅ torch `2.11.0.dev+cu130` intact |
| 256px latent cache reconstruction | ✅ 27.15 dB |
| 512px latent cache reconstruction | ✅ 30.77 dB |
| H-flip cache index alignment | ✅ 27.13 / 30.80 dB |
| Flow sampler vs. oracle denoiser | ✅ 4e-8 error, **at 1 step** |
| DDIM sampler vs. oracle denoiser | ✅ 2.4e-8 |
| CFG scale 1.0 == unguided | ✅ bit-identical |
| **DiT overfit gate (10 images, 10 species)** | ✅ **loss 1.98 → 0.0034** |
| **DiT full run, 300 epochs** | ✅ train 1.601 → 0.719, val best **0.7296** @ ep262 |
| **Attribute control (4 minimal caption pairs)** | ✅ 4/4 changed the right attribute |
| **SD 1.5 UNet download** | ✅ sha256 verified byte-exact |
| **LoRA fine-tune, 8000 steps** | ✅ val 0.1555 → 0.1500, adapter 6.4 MB |
| **Three-way comparison + CLIP scores** | ✅ see README Results |

The overfit result is in `outputs/samples/overfit_check/GATE_RESULT.png`.
That gate is the important one: it proves the objective, architecture,
conditioning, sampler, latent cache and EMA all work together.

## Measured performance (RTX 5050 Laptop, 8 GB)

- **Batch 32 is the ceiling.** Peak 5.69 GB; batch 48 OOMs.
- **122 img/s ≈ 10.4 TFLOPS bf16** — near the card's peak, so training is
  compute-bound rather than throttled by anything fixable.
- **~1.87 min/epoch** measured over the full run (349 steps/epoch at batch 32).
- LoRA: batch 8 at 512px, 4.7 GB, ~1.24 steps/s.

