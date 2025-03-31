#!/bin/bash

# ALWAYS MAKE SURE TO CHANGE TO THE RIGHT CONDA ENV before running this script
# conda activate llava-med

# edited based on LLaVA/scripts/v1_5/finetune_task_lora.sh and LLaVA/scripts/finetune_qlora.sh
# Aim: Finetune the MLP after quantizing the model

##############################################################
llava_med_dir=$HOME/LLaVA-Med
image_folder=$HOME/Datasets/mimic-cxr-jpg/2.1.0/
model_base=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2
model_name=llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2
vision_tower_path=$llava_med_dir/checkpoints/vision_tower-epoch-1-lr-0.0001
image_processor_path=$llava_med_dir/checkpoints/vision_tower-epoch-1-lr-0.0001
##############################################################
version=mistral_instruct
deepspeed_config=$llava_med_dir/scripts/zero2.json
bits=4
data_file=train_5k_custom
data_path=$HOME/Datasets/mimic-cxr/processed_data/${data_file}.json
epoch=6
freeze_backbone=True
tune_mm_mlp_adapter=True
output_dir=$llava_med_dir/checkpoints/lora-${model_name}-${data_file}-train-mlp-quantized-${bits}-epoch-${epoch}
##############################################################


# add llava directory path to PYTHONPATH so that it can be imported
export PYTHONPATH=$llava_med_dir:$PYTHONPATH
# set the max memory size to prevent memory fragmentation
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'

echo "starting the training"
echo "start time:$(date)"

deepspeed $llava_med_dir/llava/train/train.py \
    --bits $bits \
    --freeze_backbone $freeze_backbone \
    --tune_mm_mlp_adapter $tune_mm_mlp_adapter \
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
    --learning_rate 2e-4 \
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
