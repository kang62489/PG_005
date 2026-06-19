#!/bin/bash
NODES="saion-gpu15 saion-gpu16 saion-gpu18 saion-gpu19 saion-gpu21 saion-gpu07 saion-gpu08 saion-gpu09"

for node in $NODES; do
  echo -n "$node: "
  srun --immediate=5 --partition=gpu --gres=gpu:1 --nodelis=$node --time=00:01:00 \
    nvidia-smi --query-gpu=driver_version,memory.free --format=csv,noheader \
    2>/dev/null || echo "unavailable"
done
