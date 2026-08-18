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

## Findings worth keeping

**CFG 3.0, not 4.0.** Training grids were drawn at a fixed CFG of 4.0 and the
late ones looked worse than the mid-training ones. It was not degradation: the
guidance a model wants *falls* as it trains. The sweep shows CFG 1 muted, 3
crisp, 6–8 posterised into flat colour. Use 3.0.

**`last.ckpt` held epoch 262, not 299.** Lightning only wrote checkpoints on
val-loss improvement, and 262 was the last one. No real loss (262 *is* the best
val model) but the final 37 epochs are gone. For future runs pass
`every_n_epochs` / `save_on_train_epoch_end` to ModelCheckpoint if you want the
true last weights.

**Hydra `_self_` ordering.** It was listed *first* in the defaults, so
`configs/data/cub.yaml` silently overwrote every `data.*` value in the run
configs — two training launches ran at batch 64 regardless of what was set, and
OOMed identically. `_self_` must come **last**. Fixed in all four configs.

**`model=` collides with the `configs/model/` group.** The sample config key is
now `model_kind`.

## Superseded: previously running

```
scripts/train.py --config-name train_dit run_name=dit_small
```

- Detached (`setsid nohup`), so it survives the terminal closing.
- Log: `logs/dit_small_run.log`
- Budget: 120,000 steps ≈ 343 epochs ≈ **8.7 hours**
- Sample grids every 10 epochs → `outputs/samples/dit_small/`
- Checkpoints every 10 epochs → `checkpoints/dit_small/` (~900 MB each, keeps 2 + last)

Watch it:

```bash
tensorboard --logdir logs/                 # curves + sample-grid images
tail -f logs/dit_small_run.log
python scripts/plot_curves.py logs/dit_small
```

Stop it (note the bracket — it stops pkill matching its own shell):

```bash
pkill -f "scripts/train.p[y]"
```

Resume from a checkpoint:

```bash
python scripts/train.py --config-name train_dit run_name=dit_small \
  ckpt_path=checkpoints/dit_small/last.ckpt
```

## Blocked on network

**SD 1.5 LoRA fine-tune.** Its UNet download reached ~516 MB of 1.72 GB.
The partial blob is preserved in the HF cache, so reconnecting resumes rather
than restarting. When back online:

```bash
python scripts/train.py --config-name train_lora
```

Everything else it needs (VAE, CLIP text encoder, 512px latent cache) is
already local.

## Working fully offline

Set this to stop HuggingFace stalling on network timeouts when it tries to
check for updated files:

```bash
export HF_HUB_OFFLINE=1
```

Verified offline-capable: CLIP text encoder, SD VAE, both latent caches, the
DiT, TensorBoard/CSV logging, and the Gradio app (`share=False`, localhost only).

## Not yet done

- **GUI end-to-end test.** `app.py` is written and both checkpoints now exist,
  but it has never been launched against them. Run `python app.py` — it should
  offer all three models and a side-by-side Compare tab.
- Optional: a longer LoRA run (this one stopped at 8,000 steps to leave time for
  evaluation), or `model=dit_base` (111M) for the from-scratch side.

## Traps worth remembering

- `pgrep`/`pkill -f "pattern"` **matches its own shell** when the pattern
  appears in the command line. Cost about 25 minutes here. Use a bracket:
  `"scripts/train.p[y]"`.
- Tiny datasets need `overfit_repeat`. A 10-image epoch is one optimiser step,
  and Lightning's per-epoch overhead made it 2.1 s/epoch at 1% GPU use.
- `from_pretrained` without `variant="fp16"` downloads the fp32 weights
  (3.44 GB vs 1.72 GB) and casts them down on load.
- Latent caches are fp16 on disk, so encoding them in fp32 buys precision that
  is discarded at write time. bf16 costs −0.04 dB and is 1.8x faster.
