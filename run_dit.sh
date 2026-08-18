#!/bin/bash
# Unattended launcher for the from-scratch DiT run.
#
# Tries batch 32 first, and falls back to 24 then 16 if the GPU cannot hold it.
# The point is that an overnight run should never end up dead just because the
# batch size was one notch too ambitious -- only a genuine OOM triggers a
# fallback; any other failure stops and reports.
cd /home/baraa-lazkani/text_to_image || exit 1

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
PY=/home/baraa-lazkani/tiny_cd/venv/bin/python
LOG=logs/dit_small_run.log

for BS in 32 24 16; do
  echo "=== attempting batch_size=$BS ===" >> "$LOG"
  rm -rf logs/dit_small checkpoints/dit_small outputs/samples/dit_small
  "$PY" -u scripts/train.py --config-name train_dit run_name=dit_small \
        data.batch_size="$BS" sampling.every_n_epochs=2 >> "$LOG" 2>&1
  code=$?

  if [ $code -eq 0 ]; then
    echo "=== finished cleanly at batch $BS ===" >> "$LOG"
    exit 0
  fi

  if ! tail -300 "$LOG" | grep -q "OutOfMemoryError"; then
    echo "=== stopped at batch $BS (not an OOM), exit $code ===" >> "$LOG"
    exit $code
  fi

  echo "=== batch $BS hit OOM, falling back ===" >> "$LOG"
done

echo "=== all batch sizes OOMed ===" >> "$LOG"
exit 1
