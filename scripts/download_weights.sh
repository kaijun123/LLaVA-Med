conda activate llava-med

cd $HOME/LLaVA-Med/checkpoints

huggingface-cli download kaijun123/fyp llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001/model-00001-of-00004.safetensors --local-dir .
huggingface-cli download kaijun123/fyp llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001/model-00002-of-00004.safetensors --local-dir .
huggingface-cli download kaijun123/fyp llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001/model-00003-of-00004.safetensors --local-dir .
huggingface-cli download kaijun123/fyp llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001/model-00004-of-00004.safetensors --local-dir .
