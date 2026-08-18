#!/usr/bin/env python
"""Render static PNG training curves from a run's CSV log.

    python scripts/plot_curves.py logs/dit_small
    python scripts/plot_curves.py logs/dit_small --out docs/figures/dit_loss.png

TensorBoard is the interactive view; this exists so curves can be dropped into
the README without asking anyone to start a server. Uses matplotlib, which the
environment already has -- no extra dependency.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless box
import matplotlib.pyplot as plt  # noqa: E402


def find_metrics_csv(run_dir: Path) -> Path:
    """Locate metrics.csv, including inside Lightning's version_N folders."""
    direct = run_dir / "metrics.csv"
    if direct.exists():
        return direct

    versions = sorted(run_dir.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not versions:
        raise FileNotFoundError(f"no metrics.csv under {run_dir}")
    return versions[-1]


def read_series(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    """Parse the CSV into {metric: (steps, values)}, skipping blank cells.

    Lightning writes one row per logging event with most columns empty, so each
    metric has to be gathered independently rather than read as a dense table.
    """
    series: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))

    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            step = row.get("step") or row.get("epoch")
            if not step:
                continue
            for key, value in row.items():
                if key in ("step", "epoch") or not value:
                    continue
                try:
                    series[key][0].append(float(step))
                    series[key][1].append(float(value))
                except ValueError:
                    continue
    return dict(series)


def plot(series: dict, out_path: Path, title: str) -> Path:
    losses = {k: v for k, v in series.items() if "loss" in k.lower()}
    others = {k: v for k, v in series.items() if k not in losses and k != "ema_decay"}

    panels = [("Loss", losses)] + ([("Other metrics", others)] if others else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.5), squeeze=False)

    for axis, (label, group) in zip(axes[0], panels):
        for name, (steps, values) in sorted(group.items()):
            axis.plot(steps, values, label=name, linewidth=1.4)
        axis.set_xlabel("step")
        axis.set_ylabel(label.lower())
        axis.set_title(label)
        axis.grid(alpha=0.3)
        if group:
            axis.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="e.g. logs/dit_small")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    csv_path = find_metrics_csv(args.run_dir)
    series = read_series(csv_path)
    if not series:
        raise SystemExit(f"no numeric metrics found in {csv_path}")

    out = args.out or args.run_dir / "curves.png"
    plot(series, out, title=args.run_dir.name)
    print(f"read {csv_path}")
    print(f"metrics: {', '.join(sorted(series))}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
