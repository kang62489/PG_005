#!/bin/bash
# Scan every node of the gpu partition (list from sinfo, not hard-coded); try a 1-min job only on idle / mix nodes.
# Usage: . find_gpu_node.sh   (or: bash find_gpu_node.sh)

for line in $(sinfo -h -N -p gpu -o "%N:%T" | sort -u); do
  node=${line%%:*}
  state=${line#*:}
  echo -n "$node [$state]: "
  case $state in
    idle*|mix*)
      srun --immediate=5 --partition=gpu --gres=gpu:1 --nodelist="$node" --time=00:01:00 \
        nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader \
        2>/dev/null || echo "unavailable"
      ;;
    *)
      echo "skipped"
      ;;
  esac
done
