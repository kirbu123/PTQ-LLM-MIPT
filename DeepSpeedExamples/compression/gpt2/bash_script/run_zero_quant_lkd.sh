#!/bin/bash
##################fine-tune the origin model and then apply zeroquant, the following command will take approximately 10 mins in A100
###zero-quant https://arxiv.org/abs/2206.01861

# Set root path
export PYTHONPATH="/home/buka2004/PTQ-LLM-MIPT:$PYTHONPATH"

CONFIG=/home/buka2004/PTQ-LLM-MIPT/DeepSpeedExamples/compression/gpt2/config/ds_config_W8A8_Qgroup64_fp32.json
SAVE_PATH=/home/buka2004/PTQ-LLM-MIPT/DeepSpeedExamples/compression/gpt2/out/ZeroQuant/W8A8_quantization_lkd
mkdir -p ${SAVE_PATH}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%

# Set gpt-based model
MODEL=openai-community/gpt2-large
DEVICE=0
dataset_config_name=wikitext-2-raw-v1

######### fp16

source /home/buka2004/PTQ-LLM-MIPT/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=1,3,7

# Disturbed launch

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12345 \
    -m DeepSpeedExamples.compression.gpt2.run_clm_lkd \
    --dataset_name wikitext \
    --dataset_config_name ${dataset_config_name} \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --deepspeed_config ${CONFIG} \
    --deepspeed \
    --device ${DEVICE} \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --smooth \
    --smooth_output_path ${SAVE_PATH}/act_scales/gpt2-large.pt \
    --smooth_dataset_path /home/buka2004/PTQ-LLM-MIPT/smoothquant/datasets/val.jsonl.zst \
    --num_samples 1024 \
    --seq_len 128 \
    --alpha 0.5 \
    --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log
