#!/usr/bin/env python
"""Show training progress at a glance.

    python scripts/status.py                 # one snapshot
    python scripts/status.py --watch         # refresh every 30s
    python scripts/status.py --run sd15_lora_cub

Read-only: it inspects log files, checkpoints and nvidia-smi. It never touches
the training process, so it is always safe to run against a live job.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import timedelta
from pathlib import Path

BAR_WIDTH = 40


def sh(cmd: str) -> str:
    """Run a fixed diagnostic command and return its stdout.

    ``shell=True`` is used because these are pipelines (pgrep | head, df | awk).
    Every call site passes a hardcoded literal -- no argument, filename or user
    input is ever interpolated into these strings, so there is no injection
    surface. Keep it that way: if you ever need to pass a value in, switch that
    call to a list form without a shell.
    """
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def find_metrics(run: str) -> Path | None:
    candidates = sorted(Path("logs").glob(f"{run}/version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    direct = Path(f"logs/{run}/metrics.csv")
    if direct.exists():
        candidates.append(direct)
    return candidates[-1] if candidates else None


def read_metrics(path: Path) -> dict:
    """Pull the last value of each metric, plus the epoch/step position."""
    latest: dict[str, tuple[int, float]] = {}
    epoch = step = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = max(epoch, int(row.get("epoch") or 0))
                step = max(step, int(row.get("step") or 0))
            except ValueError:
                pass
            for key, value in row.items():
                if key in ("epoch", "step") or not value:
                    continue
                try:
                    latest[key] = (epoch, float(value))
                except ValueError:
                    pass
    return {"epoch": epoch, "step": step, "metrics": latest}


def loss_history(path: Path, key: str, points: int = 10) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(key):
                try:
                    series.append((int(row["epoch"]), float(row[key])))
                except ValueError:
                    pass
    if len(series) <= points:
        return series
    stride = len(series) // points
    return series[::stride][-points:]


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    return "".join(blocks[min(int((v - low) / span * (len(blocks) - 1)), len(blocks) - 1)] for v in values)


def render(run: str, total_epochs: int) -> str:
    out: list[str] = []
    add = out.append

    alive = bool(sh('pgrep -f "scripts/train.p[y]"'))
    uptime = sh("ps -o etime= -p $(pgrep -f 'scripts/train.p[y]' | head -1) 2>/dev/null")
    add("=" * 62)
    add(f"  {run}   {'RUNNING' if alive else 'NOT RUNNING'}" + (f"   up {uptime.strip()}" if uptime else ""))
    add("=" * 62)

    metrics_path = find_metrics(run)
    if not metrics_path:
        add("  no metrics yet -- training may still be starting up")
        return "\n".join(out)

    state = read_metrics(metrics_path)
    epoch, step = state["epoch"], state["step"]

    done = epoch / total_epochs if total_epochs else 0
    filled = int(done * BAR_WIDTH)
    add(f"  epoch {epoch}/{total_epochs}   step {step:,}")
    add(f"  [{'#' * filled}{'.' * (BAR_WIDTH - filled)}] {done * 100:5.1f}%")

    # Rate and ETA from the training process's own elapsed time. (Not the log
    # file's ctime -- that updates on every append, so it always reads as new.)
    elapsed_s = sh("ps -o etimes= -p $(pgrep -f 'scripts/train.p[y]' | head -1) 2>/dev/null")
    if elapsed_s.isdigit() and epoch > 0:
        elapsed = int(elapsed_s)
        per_epoch = elapsed / max(epoch, 1)
        remaining = (total_epochs - epoch) * per_epoch
        add(f"  {per_epoch / 60:.2f} min/epoch   elapsed {timedelta(seconds=elapsed)}"
            f"   ETA {timedelta(seconds=int(remaining))}  (finishes ~"
            f"{time.strftime('%H:%M', time.localtime(time.time() + remaining))})")

    add("")
    for key in ("train/loss_epoch", "val/loss"):
        if key in state["metrics"]:
            history = loss_history(metrics_path, key)
            values = [v for _, v in history]
            first = values[0] if values else float("nan")
            add(f"  {key:18s} {first:.4f} -> {values[-1]:.4f}  {sparkline(values)}")

    grids = sorted(Path(f"outputs/samples/{run}").glob("epoch_*.png"))
    if grids:
        add("")
        add(f"  sample grids: {len(grids)}   latest: {grids[-1]}")

    ckpts = sorted(Path(f"checkpoints/{run}").glob("*.ckpt"))
    if ckpts:
        size = sum(c.stat().st_size for c in ckpts) / 1e9
        add(f"  checkpoints : {len(ckpts)} ({size:.1f} GB)")

    gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader")
    disk = sh("df -h /home | tail -1 | awk '{print $4\" free\"}'")
    add("")
    add(f"  gpu: {gpu}    disk: {disk}")
    add("=" * 62)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="dit_small")
    parser.add_argument("--epochs", type=int, default=300, help="total epochs, for the progress bar")
    parser.add_argument("--watch", action="store_true", help="refresh continuously")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    if not args.watch:
        print(render(args.run, args.epochs))
        return

    try:
        while True:
            print("\033[2J\033[H" + render(args.run, args.epochs))
            print(f"\n  refreshing every {args.interval}s -- Ctrl-C to exit")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching (training is unaffected)")


if __name__ == "__main__":
    main()
