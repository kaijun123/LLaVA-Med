#!/bin/bash

llava_med_dir=$HOME/LLaVA-Med
export PYTHONPATH=$llava_med_dir:$PYTHONPATH

# model_base=microsoft/llava-med-v1.5-mistral-7b

model_name=llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k_custom-train-mlp-unquantized-epoch-1-lr-6e-5-mm_projector_lr-6e-5
model_base=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2
lora_adapter=$llava_med_dir/checkpoints/lora-llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k_custom-train-mlp-unquantized-epoch-1-lr-6e-5-mm_projector_lr-6e-5
save_path=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k_custom-train-mlp-unquantized-epoch-1-lr-6e-5-mm_projector_lr-6e-5
vision_tower_path=$llava_med_dir/checkpoints/vision_tower-epoch-1-lr-0.0001
image_processor_path=$llava_med_dir/checkpoints/vision_tower-epoch-1-lr-0.0001

python $llava_med_dir/scripts/merge_lora_weights.py \
  --model-name $model_name \
  --model-path $lora_adapter \
  --model-base $model_base \
  --save-model-path $save_path \
  --vision_tower_path $vision_tower_path \
  --image_processor_path $image_processor_path
