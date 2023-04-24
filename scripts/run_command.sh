#!/bin/bash
echo "Running train command for similar environments..."
python nn/train.py  --environment same --train_for material 

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py --test_results 1

echo "Running train command for different environments..."
python nn/train.py --environment different --train_for material 

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py --test_results 1

echo "Running train command for similar environments object wise..."
python nn/train.py  --environment same --train_for Y 

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py --test_results 1

echo "Running train command for different environments object wise..."
python nn/train.py --environment different --train_for Y 

echo "Running evaluation metrics..."
python metrics/compute_aps_read_name.py --test_results 1