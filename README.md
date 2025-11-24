# PTQ-LLM-MIPT

Post-Training Quantization for Large Language Models - MIPT Research Project

## Project Overview

This project implements and experiments with Post-Training Quantization (PTQ) techniques for Large Language Models, combining DeepSpeed and SmoothQuant methodologies.

Welcome to our: [literary review](https://docs.google.com/spreadsheets/d/1vHBZKW7IKO7Z1W8Cb-9dAWTvs5KyeQz7na4ITVk3UbE/edit?usp=sharing)

## Installation

### Clone the repository

```bash
git clone https://github.com/kirbu123/PTQ-LLM-MIPT.git
cd PTQ-LLM-MIPT
```

### Setup enviroment

```bash
# python --version = 3.8 for DeepSpeed and 3.10 for vllm pipeline
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ./llm-compressor --upgrade -r requirements.txt
```

### Usage example

```bash
python notebooks/do_compression.py \
                    --device cuda \
                    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
                    --dataset_name "wikitext" \
                    --dataset_subset "wikitext-2-raw-v1" \
                    --scheme "W8A8" \
                    --targets "Linear" \
                    --next_reg_lam 0.1 \
                    --num_calibration_samples 1024 \
                    --max_seq_length 1024 \
                    --seed 42 \
                    --output_dir "quant_checkpoints/smoothquant+gptq" \
                    --smoothing_strength 0.8 \
                    --smoothquant \
                    --gptq
```
