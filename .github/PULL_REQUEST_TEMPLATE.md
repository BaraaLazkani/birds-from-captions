## What this changes

A short description of the change and why it is needed.

## Type

- [ ] Bug fix
- [ ] New feature or experiment
- [ ] Documentation
- [ ] Refactor (no behaviour change)

## Verification

- [ ] `python tests/test_sampler.py` passes
- [ ] Data pipeline changes: reconstruction PSNR still above 20 dB
- [ ] Model or objective changes: the 10-image overfit gate still converges
- [ ] Memory: measured peak VRAM in the real training loop, not in isolation

## Results

If this changes generation quality, include before/after sample grids at a
**fixed seed and fixed guidance scale** — otherwise the comparison is not
controlled.
