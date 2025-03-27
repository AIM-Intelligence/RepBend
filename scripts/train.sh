#!/bin/bash

export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

METHOD='rep_bending' # NPO, DPO, rmu, WHP, WHP_retrain, circuit_breaker, task_arithmetic, rep_bending
DEVICE=0 # For AWS server, use one of GPUs: 0,1,2,3,4,5,6,7
MODEL="Meta_Llama3_8b" # Mistral-7b, Meta_Llama3_8b

if [ $MODEL == "Mistral-7b" ]; then
    model_name_or_path=mistralai/Mistral-7B-Instruct-v0.2
elif [ $MODEL == "Meta_Llama3_8b" ]; then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
fi

# set hyperparameters based on the method
if [ $METHOD == "NPO" ]; then
    learning_rate=5e-5
    alpha=1.0
    beta=0.1
    gamma=0.1
    max_step=1000
    eval_steps=20000
    max_seq_length=4096
    per_device_train_batch_size=4
    gradient_accumulation_steps=4
elif [ $METHOD == "DPO" ]; then
    learning_rate=5e-5
    beta=0.1
    eval_steps=20000
    max_seq_length=4096
    per_device_train_batch_size=16
    gradient_accumulation_steps=1
elif [ $METHOD == "WHP" ]; then
    learning_rate=5e-5
    alpha=0.5
    eval_steps=20000
    max_seq_length=4096
    per_device_train_batch_size=16
    gradient_accumulation_steps=1
elif [ $METHOD == "WHP_retrain" ]; then
    learning_rate=5e-5
    alpha=5.0
    beta=0.5
    gamma=0.0
    eval_steps=20000
    max_seq_length=4096
    per_device_train_batch_size=16
    gradient_accumulation_steps=1
elif [ $METHOD == "circuit_breaker" ]; then
    learning_rate=1e-4
    alpha=10.0
    beta=0.0
    gamma=0.0
    epsilon=0.0
    eta=0.0
    target_layers="18,29,20,21,22,23,24,25,26,27,28,29,30"
    transform_layers="-1"
    max_step=150
    dataset_mode=1
    eval_steps=1000
    max_seq_length=2048
    per_device_train_batch_size=4
    gradient_accumulation_steps=4
elif [ $METHOD == "task_arithmetic" ]; then
    learning_rate=5e-5
    alpha=0.5
    eval_steps=20000
    max_seq_length=4096
    per_device_train_batch_size=16
    gradient_accumulation_steps=1
elif [ $METHOD == "rep_bending" ]; then
    learning_rate=1e-5
    alpha=0.5
    beta=0.5
    gamma=0.1
    epsilon=0.3
    eta=0.0
    target_layers="-1"
    target_layer_start_idx=20
    layers_window_size=11
    transform_layers="-1"
    max_step=450
    loss_mode="response_all"
    alpha_mode="all"
    eval_steps=1000
    max_seq_length=4096
    per_device_train_batch_size=4
    gradient_accumulation_steps=4
elif [ $METHOD == "rmu" ]; then
    learning_rate=5e-5
    alpha=1200
    steering_coeff=6.5
    max_step=150
    dataset_mode=1
    layer_ids="5,6,7"
    layer_id=7
    eval_steps=20000
    max_seq_length=4096
    per_device_train_batch_size=16
    gradient_accumulation_steps=1
    param_ids="6"
fi

output="${MODEL}_${METHOD}_lr${learning_rate}"

# set output name based on the method
if [ $METHOD == "NPO" ]; then
    output="${output}_alpha${alpha}_beta${beta}_gamma${gamma}_maxstep${max_step}"
elif [ $METHOD == "DPO" ]; then
    output="${output}_beta${beta}"
elif [ $METHOD == "WHP" ]; then
    output="${output}_alpha${alpha}"
elif [ $METHOD == "WHP_retrain" ]; then
    output="${output}_alpha${alpha}_beta${beta}_gamma${gamma}"
elif [ $METHOD == "circuit_breaker" ]; then
    output="${output}_layers${target_layers}_${alpha}_${max_step}step_${dataset_mode}"
elif [ $METHOD == "task_arithmetic" ]; then
    output="${output}_alpha${alpha}"
elif [ $METHOD == "rep_bending" ]; then
    output="${output}_target_layers${target_layers}_${target_layer_start_idx}_num${layers_window_size}_${alpha}_${beta}_${gamma}_${epsilon}_${eta}_${max_step}step_${loss_mode}_${alpha_mode}"
elif [ $METHOD == "rmu" ]; then
    output="${output}_rmu_layers${layers}_unlearn${layer_id}_${alpha}_${steering_coeff}_${max_step}step_${dataset_mode}"
fi

output_dir="./out/${output}"

echo "model_name_or_path=$model_name_or_path"
echo "user_tag=$user_tag"
echo "assistant_tag=$assistant_tag"
echo "output_dir=$output_dir"
# set default hyperparameters for not specified hyperparameters

# --target_layers $layers \
CUDA_VISIBLE_DEVICES=$DEVICE \
accelerate launch --config_file configs/accelerate_zero1.yaml \
    --num_processes 1 --main_process_port $MASTER_PORT --deepspeed_hostfile ds_hostfile \
    methods/$METHOD/train.py \
    --model_name_or_path $model_name_or_path \
    --dataset_path "allenai/wildguardmix" \
    --dataset_split "wildguardtrain" \
    ${target_layers:+--target_layers $target_layers} \
    ${target_layer_start_idx:+--target_layer_start_idx $target_layer_start_idx} \
    ${layers_window_size:+--layers_window_size $layers_window_size} \
    ${transform_layers:+--transform_layers $transform_layers} \
    ${alpha:+--loss_alpha $alpha} \
    ${beta:+--loss_beta $beta} \
    ${gamma:+--loss_gamma $gamma} \
    ${epsilon:+--loss_epsilon $epsilon} \
    ${eta:+--loss_eta $eta} \
    ${loss_mode:+--loss_mode $loss_mode} \
    ${alpha_mode:+--alpha_mode $alpha_mode} \
    ${dataset_mode:+--dataset_mode $dataset_mode} \
    ${max_step:+--max_steps $max_step} \
    ${steering_coeff:+--steering_coeff $steering_coeff} \
    ${layer_id:+--layer_id $layer_id} \
    ${layer_ids:+--layer_ids $layer_ids} \
    ${param_ids:+--param_ids $param_ids} \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    ${output_dir:+--output_dir $output_dir} \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps $eval_steps \
    --save_total_limit 0 \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --max_seq_length $max_seq_length \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --is_online False

# EVALUATION
# "allenai/wildguardmix" 2048


# cd /home/ashkan/safety-eval
# # python3 -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect(); torch.cuda.empty_cache()"

# TASKS=("harmbench:harmbench_classifier" "wildguardtest:harmbench_classifier" "xstest")
# MODEL_DICT=(
#     # "mistral_7B|/home/ashkan/safety-analysis/outs/${output}|${output}|mistral"
#     "llama3_8B|/home/ashkan/safety-analysis/outs/${output}|${output}|llama3"
# )

# # Loop over each task
# for TASK in "${TASKS[@]}"; do
#     # Loop over each model and task
#     for model_info in "${MODEL_DICT[@]}"; do
#         # Split the model info string into individual variables
#         IFS='|' read -r MODEL_TYPE MODEL_PATH MODEL_NAME TEMPLATE <<< "$model_info"

#         echo "Running eval.sh with Model: $MODEL_NAME, Task: $TASK"

#         # Submit the eval.sh script with the parameters
#         sbatch eval.sh "$MODEL_TYPE" "$MODEL_PATH" "$MODEL_NAME" "$TEMPLATE" "$TASK"
#     done
# done
