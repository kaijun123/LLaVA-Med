## FYP Guide

### Setup Guide
The following steps are copied from the README. Refer to the README for more details if necessary.

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/microsoft/LLaVA-Med.git
cd LLaVA
```

2. Install Package
```bash
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```


### Dependency Issues
- During the course of implementation, various dependency issues were faced.
- `llava-med.yaml` provides a configuration which worked for this project
- Note: cuda-12.4 was used

### Code Intro:
- `code/utils.py`: contains utility code
- `code/validate.py`: code to obtain the generated output for evaluation of model performance
- `code/merge_lora_weights.py`: merge lora weights with pretrained weights
- `code/configure.py`: contains code to replace the vision tower of a pretrained model with a finetuned vision encoder

- `scripts`: Contains bash scripts and slurm jobscripts for reference. Edit the paths whenever necessary. slurm jobscripts end with the suffix `jobscript`
  - `configure*`: Executes `code/configure.py` to replace the vision tower of a pretrained model
  - `finetune_lora*`: scripts to finetune the mlp and llm without quantization
  - `finetune_mlp*`: scripts to finetune the mlp without quantization
  - `finetune_qlora_mlp*`: scripts to finetune the mlp with quantization
  - `finetune_qlora_llm*`: scripts to finetune the mlp and llm with quantization
  - `merge_weights*`: merge pretrained weights with lora weights
  - `validate*`: Executes `code/validate.py` to generate the output from the model
