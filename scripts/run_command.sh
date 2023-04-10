#!/bin/bash
echo "Running train command for similar environments..."
python nn/train.py --sample_limit 10 --reps 1

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py

# echo "Running train command for different environments..."
# python nn/train.py --environment different

# echo "Running evaluation metrics..."
# python metrics/computcompute_aps_read_namee_aps.py 