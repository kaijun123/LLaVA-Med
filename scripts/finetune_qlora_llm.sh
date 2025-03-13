#!/bin/bash

# NOTE: this bash script is running llava code using llava-med weights, and our own dataset
# ALWAYS MAKE SURE TO CHANGE TO THE RIGHT CONDA ENV before running this script
# conda activate llava

# edited based on LLaVA/scripts/v1_5/finetune_task_lora.sh and LLaVA/scripts/finetune_qlora.sh

##############################################################
llava_dir=$HOME/LLaVA
llava_med_dir=$HOME/LLaVA-Med
image_folder=$HOME/physionet.org/files/mimic-cxr-jpg/2.1.0/
model_base=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k-train-mlp-quantized-4-epoch-6
##############################################################
# changed params: bits (quantization), deepspeed config (zero2), conv mode (mistral_instruct)
version=mistral_instruct
deepspeed_config=$llava_dir/scripts/zero2.json
bits=4
data_file=train_5k
data_path=$HOME/MIMIC-CXR/processed_data/${data_file}.json
epoch=1
lr=2e-5
output_dir=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k-train-mlp-quantized-4-epoch-6-${data_file}-train-mlp-llm-quantized-${bits}-epoch-${epoch}-lr-${lr}
##############################################################


# add llava directory path to PYTHONPATH so that it can be imported
export PYTHONPATH=$llava_dir:$PYTHONPATH
# set the max memory size to prevent memory fragmentation
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'

echo "starting the training"
echo "start time:$(date)"

deepspeed $llava_med_dir/llava/train/train.py \
    --bits $bits \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed $deepspeed_config \
    --model_name_or_path $model_base \
    --version $version \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower $model_base \
    --vision_tower_path $model_base \
    --image_processor_path $model_base \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

echo "end time:$(date)"
