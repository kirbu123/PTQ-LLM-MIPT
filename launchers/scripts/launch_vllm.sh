#!/bin/bash

# Constants - Update these as needed
DEVICE="cuda:1"
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME="wikitext"
DATASET_SUBSET="wikitext-2-raw-v1"
SCHEME="W8A8"
TARGETS="Linear"
NEXT_REG_LAM=0.0
NEXT_LOSS_LAM=0.1
NUM_CALIBRATION_SAMPLES=8
MAX_SEQ_LENGTH=8
SEED=42 # Important to remain constant
SMOOTHING_STRENGTH=0.5
HES_REG_LAM=0.0
KERNEL_MODE='gaussian'

# Paths (relative to launchers/scripts/)
COMPRESSION_SCRIPT="../do_compression.py"
OUTPUT_DIR="../../quant_checkpoints/test"

# Error handling
set -e

echo "Starting compression script..."

# Check if we're in the correct directory
if [ ! -f "../do_compression.py" ]; then
    echo "Error: do_compression.py not found at ${COMPRESSION_SCRIPT}"
    echo "Please run this script from launchers/scripts/"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Set cuda
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# Run the compression script
python "${COMPRESSION_SCRIPT}" \
    --device "${DEVICE}" \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_subset "${DATASET_SUBSET}" \
    --scheme "${SCHEME}" \
    --targets "${TARGETS}" \
    --num_calibration_samples "${NUM_CALIBRATION_SAMPLES}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --smoothing_strength "${SMOOTHING_STRENGTH}" \
    --smoothquantreg \
    --hes_reg_lam "${HES_REG_LAM}" \
    --gptq \
    --next_reg_lam "${NEXT_REG_LAM}" \
    --next_loss_lam "${NEXT_LOSS_LAM}" \
    --kernel_mode "${KERNEL_MODE}"

echo "Compression script completed successfully!"
echo "Output saved to: ${OUTPUT_DIR}"