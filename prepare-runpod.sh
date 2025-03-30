#!/usr/bin/env bash

apt update
apt install unzip vim

dir=$(pwd)
python -m venv venv
source "$dir/venv/bin/activate"

echo "export HF_HOME=/workspace/caches/huggingface\n" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=/workspace/caches/transformers\n" >> ~/.bashrc
echo "export TORCH_HOME=/workspace/caches/torch\n" >> ~/.bashrc

source ~/.bashrc
