#!/bin/bash

# Constants - Update these as needed
DEVICE="cuda:1"
ACTION="${1:-gptq}"

if [[ "${ACTION}" != "gptq" && "${ACTION}" != "dptq" && "${ACTION}" != "awq" && "${ACTION}" != "rtn" ]]; then
    echo "Invalid action: ${ACTION}. Use 'gptq', 'dptq', 'awq', or 'rtn'."
    exit 1
fi

MODEL_BASE_PATH="/home/buka2004/data/weights/"
DATASET_BASE_PATH="/home/buka2004/data/datasets/"

# Large models
# MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME="facebook/opt-125m"
# MODEL_NAME="facebook/opt-1.3b"
# MODEL_NAME="Qwen/Qwen2-0.5B"

# Small models
# MODEL_NAME="facebook/opt-125m"
# MODEL_NAME="cerebras/Cerebras-GPT-111M"
# MODEL_NAME="ComCom/gpt2-small"

# Pythia models
# MODEL_NAME="EleutherAI/gpt-neo-125m"
# MODEL_NAME="EleutherAI/pythia-14m-deduped"
# MODEL_NAME="EleutherAI/pythia-31m"
# MODEL_NAME="EleutherAI/pythia-70m"
# MODEL_NAME="EleutherAI/pythia-160m"
# MODEL_NAME="openai-community/gpt2"
# MODEL_NAME="facebook/opt-6.7b"

DATASET_NAME="wikitext"
DATASET_SUBSET="wikitext-2-raw-v1"
SCHEME="W4A8"
TARGETS="Linear"
NUM_CALIBRATION_SAMPLES=1024
MAX_SEQ_LENGTH=1024
SEED=0
LAM_LR=3e-4
K_NEXT=6
opt_steps_num=1000 # set 1 for debug !!!
SMOOTHING_STRENGTH=0.5
HES_REG_LAM=0.0
NEXT_REG_LAM=0.0 # for multistep: 0.0
NEXT_LOSS_LAM=0.0
KERNEL_MODE="default"
LAM_OPTIMIZE_METHOD="multistep"
LAM_LOSS_NAME="ElboPowerLawLossNewFast"
# ElboPowerLawLoss ElboPowerLawLossNewFast ElboPowerLawLossNew HessianLossTraceOnlyScaled ElboPowerLawLossTrunc 
# ReformulatedElboPowerLawLossTrunc HessianLossSoftCos 
# HessianLossTraceReformulatedInverse HessianLossTraceReformulated
# HessianLossTraceOnlyScaledReformulated HessianLossTraceOnlyScaledReformulatedInverse

NEXT_STRAT_NAME="AllLinears" # AllLinears BasicStrat IgnoreNotOutProj
TASKS="wikitext,hellaswag,piqa,arc_easy"

# Extended grid search parameters
GRID_VALUES=(0 10 20 30 40 50 60 70 80 90)

# Paths
COMPRESSION_SCRIPT="./launchers/do_compression.py"
OUTPUT_BASE_DIR="./quant_checkpoints/no-smooth/optimize" # no-optimize

LOG_DIR="./grid_search_logs"

# Error handling
set -e
echo "Starting comprehensive grid search for soothing_strenght..."

# Create directories
mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${LOG_DIR}"

# Set CUDA device
export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}"

# Get timestamp for unique logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/grid_search_${TIMESTAMP}.log"

QUANT_SCHEME="${SCHEME}"
if [[ "${ACTION}" == "awq" && "${SCHEME}" == "W4A8" ]]; then
    QUANT_SCHEME="W4A16_ASYM"
fi

# Function to log messages
log_message() {
    local message="$1"
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $message" | tee -a "$LOG_FILE"
}

log_message "Starting grid search"
log_message "Values to test: ${GRID_VALUES[*]}"
log_message "Model: $MODEL_NAME"
log_message "Device: $DEVICE"
log_message "Quantization action: $ACTION"
log_message "Quantization scheme: $QUANT_SCHEME"

total_experiments=${#GRID_VALUES[@]}
current_experiment=1

# Grid search loop
for grid_value in "${GRID_VALUES[@]}"; do
    log_message "=========================================="
    log_message "Experiment $current_experiment/$total_experiments - grid_value = $grid_value"
    log_message "=========================================="

    # Create unique output directory
    OUTPUT_DIR="${OUTPUT_BASE_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    # --model_base_path "${MODEL_BASE_PATH}" \
    # --dataset_base_path "${DATASET_BASE_PATH}" \

    # Select quantization backend flag
    QUANT_FLAG="--${ACTION}"

    # Run the compression script
    python "${COMPRESSION_SCRIPT}" \
        --device "${DEVICE}" \
        --model_name "${MODEL_NAME}" \
        --dataset_name "${DATASET_NAME}" \
        --dataset_subset "${DATASET_SUBSET}" \
        --scheme "${QUANT_SCHEME}" \
        --targets "${TARGETS}" \
        --num_calibration_samples "${NUM_CALIBRATION_SAMPLES}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --seed "${grid_value}" \
        --output_dir "${OUTPUT_DIR}" \
        --hes_reg_lam "${HES_REG_LAM}" \
        --lam_lr "${LAM_LR}" \
        --k_next "${K_NEXT}" \
        --opt_steps_num "${opt_steps_num}" \
        --lam_loss_name "${LAM_LOSS_NAME}" \
        --next_strat_name "${NEXT_STRAT_NAME}" \
        ${QUANT_FLAG} \
        --next_reg_lam "${NEXT_REG_LAM}" \
        --next_loss_lam "${NEXT_LOSS_LAM}" \
        --kernel_mode "${KERNEL_MODE}" \
        --lam_optimize_method "${LAM_OPTIMIZE_METHOD}" \
        --tasks "${TASKS}" \
        --lam_optimize \
        # --smoothquant \
        # --do_hessian_plot \
        # --reinitialize_lam \

    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_message "✓ SUCCESS: grid_value: $grid_value"
    else
        log_message "✗ FAILED: grid_value: $grid_value"
        exit 1
    fi

    log_message "Output: $OUTPUT_DIR"
    ((current_experiment++))
done

log_message "=========================================="
log_message "GRID SEARCH COMPLETED"
log_message "Tested values: ${GRID_VALUES[*]}"
log_message "Total experiments: $total_experiments"
log_message "Main log: $LOG_FILE"
log_message "=========================================="
