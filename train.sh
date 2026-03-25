#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash train.sh               # run default config(default.yaml), 5 times
#   bash train.sh 3             # run default config(default.yaml), 3 times
#   bash train.sh 3 ablation_a  # run conf/ablation_a.yaml, 3 times
#
# wandb run name is configured in conf/*.yaml:
#   wandb_name_prefix + "_" + wandb_run_index

RUN_COUNT="${1:-5}"
CONFIG_NAME="${2:-default}"

if ! [[ "$RUN_COUNT" =~ ^[0-9]+$ ]] || [ "$RUN_COUNT" -lt 1 ]; then
  echo "Error: RUN_COUNT must be a positive integer."
  exit 1
fi

for i in $(seq 1 "$RUN_COUNT"); do
  echo "[$(date '+%F %T')] Starting run ${i}/${RUN_COUNT}: CONFIG_NAME=${CONFIG_NAME}, wandb_run_index=${i}"
  python main.py --config-name "${CONFIG_NAME}" wandb_run_index="${i}"
  echo "[$(date '+%F %T')] Finished run ${i}/${RUN_COUNT}"
done

echo "All runs completed."
