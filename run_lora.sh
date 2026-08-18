#!/bin/bash
# Unattended launcher for the SD 1.5 LoRA fine-tune.
# Tries batch 8 at 512px, falling back to 4 then 2 only on a genuine OOM.
cd /home/baraa-lazkani/text_to_image || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
PY=/home/baraa-lazkani/tiny_cd/venv/bin/python
LOG=logs/lora_run.log
for BS in 8 4 2; do
  echo "=== attempting batch_size=$BS ===" >> "$LOG"
  rm -rf logs/sd15_lora_cub checkpoints/sd15_lora_cub outputs/samples/sd15_lora_cub
  "$PY" -u scripts/train.py --config-name train_lora run_name=sd15_lora_cub \
        data.batch_size=$BS sampling.every_n_epochs=1 >> "$LOG" 2>&1
  code=$?
  if [ $code -eq 0 ]; then echo "=== finished cleanly at batch $BS ===" >> "$LOG"; exit 0; fi
  if ! tail -300 "$LOG" | grep -q "OutOfMemoryError"; then
    echo "=== stopped at batch $BS (not an OOM), exit $code ===" >> "$LOG"; exit $code; fi
  echo "=== batch $BS hit OOM, falling back ===" >> "$LOG"
done
echo "=== all batch sizes OOMed ===" >> "$LOG"; exit 1
