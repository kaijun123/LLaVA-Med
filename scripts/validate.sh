#!/bin/bash

echo "running"

# edit the path to the llava directory
export llava_dir=$HOME/LLaVA
export PYTHONPATH=$llava_dir:$PYTHONPATH

python -u $llava_med_dir/code/validate.py \
  --model_path /home/FYP/angk0064/ANGK0064/checkpoints/lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5 \
  --model_base /home/FYP/angk0064/ANGK0064/checkpoints/microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-2-lr-6e-5 \
  --model_name lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5 \
  --image_processor_path /home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001 \
  --vision_tower_path /home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001 \
  --image-base-path /home/FYP/angk0064/Datasets/mimic-cxr-jpg/2.1.0 \
  --question-file /home/FYP/angk0064/Datasets/mimic-cxr/processed_data/validate_custom.json \
  --answers-file lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5.json
echo "end"