#!/bin/bash
echo "Running train command for similar environments..."
python nn/train.py 

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py

echo "Running train command for different environments..."
python nn/train.py --environment different

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py 