#!/bin/bash

llava_med_dir=$HOME/LLaVA-Med
export PYTHONPATH=$llava_med_dir:$PYTHONPATH

python configure.py
