#!/bin/bash
echo "Running commands for material wise classification\n\n" 
echo "Running train command for similar env materials..."
python nn/test.py --exp_name main_material --lim 30

echo "Running evaluation metrics for same env materials..."
python metrics/compute_aps.py --exp_name main_material --tested_results 0 

echo "Running test command for different env materials..." 
python nn/test.py --exp_name main_material_different

echo "Running evaluation metrics from different env materials..."
python metrics/compute_aps.py --exp_name main_material_different --tested_results 0 

echo "Running commands for object wise classification\n\n" 
echo "Running train command for similar env objects..."
python nn/test.py --exp_name main_objs

echo "Running evaluation metrics for same env objects..."
python metrics/compute_aps.py --exp_name main_objs --tested_results 0 

echo "Running test command for different env objects..."
python nn/test.py --exp_name main_objs_different

echo "Running evaluation metrics from different env objects..."
python metrics/compute_aps.py --exp_name main_objs_different --tested_results 0 