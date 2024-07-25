#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


MODEL="Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/home/michael/ai/projects/computer_agent/data/form/form_train.json"
EVAL_DATA="/home/michael/ai/projects/computer_agent/data/form/form_test.json"

export CUDA_VISIBLE_DEVICES=0

python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir output_qwen \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "wandb" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --dataloader_num_workers 12 \
    --use_lora
