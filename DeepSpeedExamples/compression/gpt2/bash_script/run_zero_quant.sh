#!/bin/bash
##################fine-tune the origin model and then apply zeroquant, the following command will take approximately 10 mins in A100
###zero-quant https://arxiv.org/abs/2206.01861
export CUDA_VISIBLE_DEVICES=1

CONFIG=config/ds_config_W8A8_Qgroup64_fp32.json
SAVE_PATH=./out/ZeroQuant/W8A8_quantization
mkdir -p ${SAVE_PATH}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%

# Set gpt-based model
MODEL=openai-community/gpt2-large

######### fp16
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12345 \
    run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --deepspeed_config ${CONFIG} \
    --deepspeed \
    --output_dir ${SAVE_PATH} # &>> ${SAVE_PATH}/train.log

### the following is the output of the above command
### Before converting the module COVN1D to linear and init_compression: 19.371443732303174
### Before cleaning, Epoch at 0 with Perplexity: 19.47031304212775
### After cleaning with Perplexity: 19.47031304212775

# python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port 12345 \
#     run_clm_no_trainer.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2-large \
#     --per_device_train_batch_size 4 \
#     --num_train_epochs 0 \
#     --deepspeed_config config/ds_config_W4or8A8_Qgroup64_fp32.json \
#     --deepspeed \
#     --output_dir ./output/W4or8A8
### the following is the output of the above command
### Before converting the module COVN1D to linear  and init_compression: 19.371443732303174
### Before cleaning, Epoch at 0 with Perplexity: 27.518339759793506
### After cleaning with Perplexity: 27.518339759793506


######### fp16
# python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port 12345 \
#     run_clm_no_trainer.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2-large \
#     --per_device_train_batch_size 4 \
#     --num_train_epochs 0 \
#     --deepspeed_config config/ds_config_W8A8_Qgroup64_fp16.json \
#     --deepspeed \
#     --output_dir ./output/W8A8_fp16
### the following is the output of the above command
### Before converting the module COVN1D to linear and init_compression: 19.371443732303174
### Before cleaning, Epoch at 0 with Perplexity: 19.618978642663098
### After cleaning with Perplexity: 19.789594118891802


# python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port 12346 \
#     run_clm_no_trainer.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --model_name_or_path gpt2-large \
#     --per_device_train_batch_size 4 \
#     --num_train_epochs 0 \
#     --deepspeed_config config/ds_config_W4or8A8_Qgroup64_fp16.json \
#     --deepspeed \
#     --output_dir ./output/W4or8A8_fp16
### the following is the output of the above command
### Before converting the module COVN1D to linear and init_compression: 19.371443732303174
### Before cleaning, Epoch at 0 with Perplexity: 31.62779929329135
### After cleaning with Perplexity: 32.426349685127285
