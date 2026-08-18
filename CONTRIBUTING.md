# Contributing

Thanks for your interest in this project. It began as coursework for a Neural
Networks module at Manara University, but the code is MIT-licensed and
contributions are welcome.

## Getting set up

```bash
git clone https://github.com/BaraaLazkani/birds-from-captions.git
cd birds-from-captions
pip install torch diffusers transformers accelerate peft gradio tensorboard
python scripts/prepare_data.py     # downloads CUB and builds the latent caches
```

The dataset download is about 1.2 GB and the latent caches take roughly 35
minutes to build on a mid-range GPU. They are written once and reused.

## Before you open a pull request

Run the sampler tests. They use oracle denoisers to prove both integrators are
numerically exact, and they will catch most changes that break the maths:

```bash
python tests/test_sampler.py
```

If you change anything in the data pipeline, re-run the verification gate and
confirm the reconstruction PSNR is still above 20 dB:

```bash
python scripts/prepare_data.py resolutions=[256] overwrite=true
```

If you change the model or the training objective, the fastest sanity check is
the overfit gate. A diffusion model that cannot memorise ten images has a bug:

```bash
python scripts/train.py --config-name train_dit overfit=10 run_name=overfit_check
```

## Things worth knowing

**Configuration is Hydra-composed.** `_self_` must stay **last** in every
`defaults:` list, or the config groups will silently overwrite the values in the
run config. This cost us two failed training runs before we found it.

**Memory is the binding constraint.** This project targets an 8 GB GPU. If you
raise a batch size or a resolution, measure the peak allocation rather than
assuming — and measure it in the real training loop, not in isolation. A bare
training step under-reports, because it misses the EMA shadow, the text encoder
and the framework's own overhead.

**Silent failures are the dangerous ones.** A broken latent cache, a wrong
sign in the sampler, or a mis-specified guidance scale will all produce a
smoothly falling loss curve while generating nonsense. Prefer adding a gate that
fails loudly over trusting that the loss looks fine.

## Style

Follow the surrounding code: type hints on public functions, docstrings that
explain *why* rather than restating *what*, and comments reserved for decisions
a reader would otherwise question.

## Reporting problems

Please open an issue using one of the templates, and include your GPU, its VRAM,
your PyTorch version and the full traceback.
