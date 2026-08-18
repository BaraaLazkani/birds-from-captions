---
name: Bug report
about: Something behaves incorrectly or crashes
title: ''
labels: bug
assignees: ''
---

**What happened**
A clear description of the problem.

**What you expected**
What you thought would happen instead.

**How to reproduce**
The exact command, including any Hydra overrides:

```bash
python scripts/train.py --config-name train_dit ...
```

**Environment**
- GPU and VRAM:
- PyTorch version (`python -c "import torch; print(torch.__version__)"`):
- CUDA / compute capability (`torch.cuda.get_device_capability()`):
- OS:

**Traceback**

```
paste the full traceback here
```

**Checked already**
- [ ] `python tests/test_sampler.py` passes
- [ ] The latent cache verification passed (PSNR above 20 dB)
