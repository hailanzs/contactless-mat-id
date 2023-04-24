#!/bin/bash
echo "Running commands for material wise classification\n\n" 
echo "Running train command for similar env materials..."
python nn/test.py --exp_name main_material_2023_04_23-10_35_57 

echo "Running evaluation metrics for same env materials..."
python metrics/compute_aps.py --exp_name main_material_2023_04_23-10_35_57 --tested_results 0 

echo "Running test command for different env materials..." 
python nn/test.py --exp_name main_material_different_2023_04_23-10_48_37

echo "Running evaluation metrics from different env materials..."
python metrics/compute_aps.py --exp_name main_material_different_2023_04_23-10_48_37 --tested_results 0 

echo "Running commands for object wise classification\n\n" 
echo "Running train command for similar env objects..."
python nn/test.py --exp_name main_objs_2023_04_23-11_04_11

echo "Running evaluation metrics for same env objects..."
python metrics/compute_aps.py --exp_name main_objs_2023_04_23-11_04_11 --tested_results 0 

echo "Running test command for different env objects..."
python nn/test.py --exp_name main_objs_different_2023_04_23-11_06_41

echo "Running evaluation metrics from different env objects..."
python metrics/compute_aps.py --exp_name main_objs_different_2023_04_23-11_06_41 --tested_results 0 