"""Oracle tests for the hand-written samplers.

The idea: hand each sampler a denoiser that predicts its target *perfectly*, and
check that integration recovers the original data.  Any sign error, misplaced
square root or off-by-one in the schedule shows up immediately as a large
reconstruction error, and these are exactly the bugs that are otherwise
invisible -- a broken sampler still produces plausible-looking noise, and a
broken trainer still produces a falling loss curve.

    python -m pytest tests/ -v      (or just: python tests/test_sampler.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttimg.diffusion.flow import FlowMatching  # noqa: E402
from ttimg.diffusion.sampler import sample_ddim, sample_flow, sd_alphas_cumprod  # noqa: E402
from ttimg.models.interface import Denoiser  # noqa: E402

SHAPE = (2, 4, 8, 8)
TOLERANCE = 1e-5


def _target() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(*SHAPE)


class _OracleFlow(Denoiser):
    """Predicts the exact rectified-flow velocity, v = eps - x0."""

    prediction_type = "flow"

    def __init__(self, x0: torch.Tensor) -> None:
        super().__init__()
        self.x0 = x0

    def forward(self, x_t, t, cond):
        t = t.view(-1, 1, 1, 1)
        # Batch may be doubled by classifier-free guidance.
        x0 = self.x0.repeat(x_t.shape[0] // self.x0.shape[0], 1, 1, 1)
        eps = (x_t - (1 - t) * x0) / t.clamp(min=1e-6)
        return eps - x0


class _OracleEpsilon(Denoiser):
    """Predicts the exact noise under SD's forward process."""

    prediction_type = "epsilon"

    def __init__(self, x0: torch.Tensor, alphas: torch.Tensor) -> None:
        super().__init__()
        self.x0 = x0
        self.alphas = alphas

    def forward(self, x_t, t, cond):
        index = (t.float() * 999).round().long().clamp(0, 999)
        alpha = self.alphas[index].view(-1, 1, 1, 1)
        x0 = self.x0.repeat(x_t.shape[0] // self.x0.shape[0], 1, 1, 1)
        return (x_t - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-6)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).norm() / expected.norm()).item()


def test_flow_sampler_is_exact_even_in_one_step():
    """Rectified-flow paths are straight, so Euler integration is exact.

    This is the defining property of the objective, and it is why the model
    stays usable at 20-30 steps instead of needing 50+.
    """
    x0 = _target()
    cond = torch.zeros(SHAPE[0], 77, 768)

    for steps in (1, 2, 10, 50):
        generator = torch.Generator().manual_seed(7)
        result = sample_flow(
            _OracleFlow(x0), cond, None, SHAPE, steps, 1.0, generator, "cpu", torch.float32
        )
        error = _relative_error(result, x0)
        assert error < TOLERANCE, f"{steps} steps gave relative error {error:.2e}"


def test_ddim_sampler_recovers_data():
    """DDIM should converge to the clean latent, with error falling as steps rise.

    Guards the terminal-alpha choice in particular: using alphas_cumprod[0]
    (0.99915) for the final step instead of 1.0 leaves ~2.9% residual noise in
    every sample, which this catches.
    """
    x0 = _target()
    alphas = sd_alphas_cumprod("cpu")
    cond = torch.zeros(SHAPE[0], 77, 768)

    previous = float("inf")
    for steps in (10, 50, 200):
        generator = torch.Generator().manual_seed(7)
        result = sample_ddim(
            _OracleEpsilon(x0, alphas), cond, None, SHAPE, steps, 1.0, generator, "cpu", torch.float32
        )
        error = _relative_error(result, x0)
        assert error < TOLERANCE, f"{steps} steps gave relative error {error:.2e}"
        assert error <= previous * 1.5, "error should not grow with more steps"
        previous = error


def test_guidance_scale_one_is_a_no_op():
    """CFG at scale 1.0 must reduce exactly to the unguided prediction."""
    x0 = _target()
    cond = torch.zeros(SHAPE[0], 77, 768)

    generator = torch.Generator().manual_seed(7)
    unguided = sample_flow(
        _OracleFlow(x0), cond, None, SHAPE, 10, 1.0, generator, "cpu", torch.float32
    )
    generator = torch.Generator().manual_seed(7)
    guided = sample_flow(
        _OracleFlow(x0), cond, torch.zeros_like(cond), SHAPE, 10, 1.0, generator, "cpu", torch.float32
    )
    assert torch.equal(unguided, guided)


def test_flow_noising_is_consistent_with_its_target():
    """add_noise and velocity_target must describe the same straight path.

    Walking from x_t back along the velocity by exactly t must land on x_0.
    """
    flow = FlowMatching()
    x0 = _target()
    noise = torch.randn_like(x0)
    t = torch.rand(x0.shape[0])

    x_t = flow.add_noise(x0, noise, t)
    velocity = flow.velocity_target(x0, noise)
    recovered = x_t - t.view(-1, 1, 1, 1) * velocity

    assert _relative_error(recovered, x0) < TOLERANCE


def test_logit_normal_concentrates_mid_trajectory():
    """The default timestep sampler should favour the middle of the path."""
    middle = lambda t: ((t > 0.2) & (t < 0.8)).float().mean().item()  # noqa: E731

    logit_normal = FlowMatching(timestep_sampling="logit_normal").sample_timesteps(20_000, "cpu")
    uniform = FlowMatching(timestep_sampling="uniform").sample_timesteps(20_000, "cpu")

    assert middle(logit_normal) > middle(uniform) + 0.15
    assert 0.45 < logit_normal.mean().item() < 0.55  # still centred


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS  {name}")
    print("\nall sampler tests passed")
