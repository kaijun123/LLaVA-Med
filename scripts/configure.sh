#!/bin/bash

llava_med_dir=$HOME/LLaVA-Med
export PYTHONPATH=$llava_med_dir:$PYTHONPATH

python -u $llava_med_dir/code/configure.py
