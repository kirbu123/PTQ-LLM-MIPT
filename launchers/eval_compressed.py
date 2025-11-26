import torch
import numpy as np
import math
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import os
from safetensors.torch import load_file

# Add imports for the critical model modification step
from transformers.pytorch_utils import Conv1D
from deepspeed.compression.helper import convert_conv1d_to_linear

# --- 1. Configuration ---
MODEL_PATH = "/home/buka2004/PTQ-LLM-MIPT/vllm_out/openai-community/gpt2-large/wikitext"
BASE_MODEL_NAME = "openai-community/gpt2-large" # The original HF model name
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_SEQ_LENGTH = 1024
STRIDE = 512
DEVICE = "cuda:1"

# def evaluate_perplexity(model, tokenizer, dataset, device='cuda'):
#     """
#     Calculates perplexity on a given dataset using a sliding window approach.
#     """
#     print(f'Set model to device: {device}')
#     model = model.to(device)

#     model = model.float()
#     model.eval()

#     print("Tokenizing and preparing the dataset...")
#     encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
#     seq_len = encodings.input_ids.size(1)
    
#     nlls = [] # Negative Log-Likelihoods
#     total_eval_tokens = 0
    
#     print(f"Starting perplexity calculation with stride {STRIDE}...")
#     prev_end_loc = 0
    
#     for begin_loc in tqdm(range(0, seq_len, STRIDE)):
#         end_loc = min(begin_loc + MAX_SEQ_LENGTH, seq_len)
#         trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()

#         with torch.no_grad():
#             # Look at model expected dtype
#             # expected_dtype = next(model.parameters()).dtype
#             # input_ids_converted = input_ids.to(torch.long)

#             # Infer model
#             outputs = model(input_ids)

#             # Shift so that tokens < n predict n
#             shift_logits = outputs.logits[..., :-1, :].contiguous()
#             shift_labels = target_ids[..., 1:].contiguous()
            
#             # Calculate loss only for the new tokens in this window
#             loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
#                            shift_labels.view(-1))
            
#             # Reshape loss to match sequence length
#             loss = loss.view(shift_labels.size())
            
#             # Only take loss for the new tokens (after the overlap)
#             if begin_loc > 0:
#                 # For alldebug  windows except the first, we skip the overlap
#                 loss = loss[:, (MAX_SEQ_LENGTH - STRIDE):]
            
#             # For the last window, we might have fewer tokens
#             if loss.size(1) > trg_len - 1:
#                 loss = loss[:, :trg_len - 1]
            
#             neg_log_likelihood = loss.sum()
#             num_tokens_in_loss = loss.numel()

#         nlls.append(neg_log_likelihood)
#         total_eval_tokens += num_tokens_in_loss
#         prev_end_loc = end_loc

#         if end_loc == seq_len:
#             break

#     # Calculate perplexity
#     total_nll = torch.stack(nlls).sum()
#     ppl = torch.exp(total_nll / total_eval_tokens)

#     return ppl.item()

def evaluate_perplexity(model, tokenizer, dataset, max_length=1024, device='cuda'):
    """
    Calculates perplexity by processing each text segment from the dataset.
    This is a more standard and reliable approach.
    """
    print(f'Setting model to device: {device}')
    model.to(device)
    model.eval() # Ensure model is in eval mode

    model = model.float()
    texts = [text for text in dataset["text"] if len(text.strip()) > 0]
    
    total_nll = 0.0
    total_tokens = 0
    
    print("Calculating perplexity...")
    for text in tqdm(texts):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        
        # Skip sequences that are too short for loss calculation
        if input_ids.size(1) < 2:
            continue
            
        with torch.no_grad():
            # Let the model calculate the loss internally by providing labels.
            # This is the standard Hugging Face method and is robust.
            # input_ids = input_ids.float()
            outputs = model(input_ids, labels=input_ids)

            # The returned loss is the average NLL for the sequence.
            loss = outputs.loss
            
            # To get the total NLL, we multiply the average loss by the number of tokens.
            num_tokens = input_ids.size(1)
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens

    # Calculate perplexity from the overall average negative log-likelihood
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    
    return perplexity


if __name__ == "__main__":
    set_seed(42)

    # --- 2. Load Quantized Model and Tokenizer ---
    print(f"Loading and preparing model architecture for '{BASE_MODEL_NAME}'...")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

    print("Converting Conv1D layers to Linear layers...")
    model = convert_conv1d_to_linear(model, Conv1D)

    model_weights_path = os.path.join(MODEL_PATH, "model.safetensors")
    print(f"Loading quantized weights from: {model_weights_path}")
    state_dict = load_file(model_weights_path)

    model.load_state_dict(state_dict, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Load Dataset ---
    print(f"Loading the '{DATASET_NAME}' dataset...")
    test_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    # --- 4. Run Evaluation ---
    print("Using simple perplexity calculation...")
    perplexity = evaluate_perplexity(model, tokenizer, test_dataset, device=DEVICE)

    # --- 5. Print Final Score ---
    print("\n" + "="*30)
    print(f"Evaluation Complete.")
    print(f"Model: {MODEL_PATH}")
    print(f"Perplexity on {DATASET_NAME} / {DATASET_SPLIT}: {perplexity:.4f}")
    print("="*30)
