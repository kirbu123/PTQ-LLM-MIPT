#!/bin/bash
##################fine-tune the origin model and then apply zeroquant, the following command will take approximately 10 mins in A100
###zero-quant https://arxiv.org/abs/2206.01861

CONFIG=config/ds_config_W8A8_Qgroup64_fp32.json
SAVE_PATH=./out/ZeroQuant/W8A8_quantization_lkd
mkdir -p ${SAVE_PATH}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%

# Set gpt-based model
MODEL=openai-community/gpt2-large
DEVICE=0

######### fp16


source /home/buka2004/PTQ-LLM-MIPT/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=1,3,7

# Disturbed launch

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12345 \
    run_clm_lkd.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --deepspeed_config ${CONFIG} \
    --deepspeed \
    --device ${DEVICE} \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log
