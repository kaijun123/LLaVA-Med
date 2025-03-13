#!/bin/bash

llava_med_dir=$HOME/LLaVA-Med
# model_base=microsoft/llava-med-v1.5-mistral-7b

model_name=vision_tower-epoch-1-lr-0.0001-train_5k-train-mlp-quantized-4-epoch-6
model_base=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001
lora_adapter=$llava_med_dir/checkpoints/lora-llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k-train-mlp-quantized-4-epoch-6
save_path=$llava_med_dir/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-train_5k-train-mlp-quantized-4-epoch-6


python $llava_med_dir/scripts/merge_lora_weights.py \
  --model-name $model_name \
  --model-path $lora_adapter \
  --model-base $model_base \
  --save-model-path $save_path