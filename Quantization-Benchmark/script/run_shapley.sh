#!/bin/bash

nohup bash run_shapley_src.sh 0.1 0 >> logs_scripts/run_shapley_alpha_0.1.log 2>&1 
nohup bash run_shapley_src.sh 0.3 0 >> logs_scripts/run_shapley_alpha_0.3.log 2>&1 
nohup bash run_shapley_src.sh 0.5 0 >> logs_scripts/run_shapley_alpha_0.5.log 2>&1 
nohup bash run_shapley_src.sh 0.7 0 >> logs_scripts/run_shapley_alpha_0.7.log 2>&1 
nohup bash run_shapley_src.sh 0.9 0 >> logs_scripts/run_shapley_alpha_0.9.log 2>&1 
