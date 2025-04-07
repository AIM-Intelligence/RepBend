#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH -p suma_a6000
#SBATCH --exclude=node26,node27,node28
#SBATCH --gres=gpu:1

ulimit -u 200000
source ~/.bashrc
ml purge
conda init bash
conda activate harmbench

model_path="mistralai/Mistral-7B-Instruct-v0.2"
# prefilling attack
# output_dir="./out/${model_path}_prefilling"

# python safety_evaluation/evaluate.py \
#  -m $model_path \
#  --benchmark harmbench_test.json \
#  --prefill True \
#  --output_dir $output_dir

# input embedding attack
output_dir="./out/${model_path}_softopt"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python safety_evaluation/evaluate.py \
 -m $model_path \
 --benchmark harmbench_test.json \
 --run_softopt \
 --batch_size 1 \
 --lr 0.0001 \
 --early_stop_loss 0.05 \
 --output_dir $output_dir
 

# --use_repe \
# --run_softopt \