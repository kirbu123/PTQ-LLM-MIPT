import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.pytorch_utils import Conv1D
from deepspeed.compression.helper import convert_conv1d_to_linear
from vllm import LLM, SamplingParams
import torch
from torch.utils.data import DataLoader, SequentialSampler
import math
import numpy as np
from transformers import default_data_collator, set_seed

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def evaluation(model, eval_dataloader, device):
    model.eval()
    losses = 0

    valid_batches = 0

    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        
        # Skip empty batches
        if batch['input_ids'].numel() == 0:
            continue
            
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss

        if loss is not None:
            valid_batches += 1
            losses += loss.float()

    if valid_batches == 0:  # No valid batches processed
        return float("inf"), float("inf")

    losses = losses / (valid_batches + 1)
    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")
    return perplexity, losses.item()

def tokenize_function(examples):
    # Filter out empty texts and tokenize
    examples["text"] = [text for text in examples["text"] if text and text.strip()]
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding=True)

def tokenize_function(examples):
    # Filter out empty texts and tokenize
    examples["text"] = [text for text in examples["text"] if text and text.strip()]
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding='max_length')

if __name__ == "__main__":
    set_seed(42)

    # Set params
    # model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model_name = 'openai-community/gpt2-large'
    # model_name = 'facebook/opt-350m'
    dataset_name = 'wikitext'
    dataset_subset = 'wikitext-2-raw-v1'

    # Set variables
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = convert_conv1d_to_linear(model, Conv1D)
    dataset = load_dataset(dataset_name, dataset_subset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = f'/home/buka2004/PTQ-LLM-MIPT/vllm_out/{model_name.replace("/", "_")}_v2/{dataset_name}'

    # Prepare evaluation data
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Filter out empty examples
    def filter_empty(example):
        return len(example['input_ids']) > 0

    eval_dataset = tokenized_datasets["validation"].filter(filter_empty)
    
    if len(eval_dataset) == 0:
        raise ValueError("No valid examples in evaluation dataset after filtering empty texts!")

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=default_data_collator, 
        sampler=eval_sampler, 
        batch_size=8
    )

    print(f"Evaluation dataset size: {len(eval_dataset)}")
    print(f"Number of batches: {len(eval_dataloader)}")

    # Evaluate original model
    print("Evaluating original model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = AutoConfig.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
    ).to(device)

    original_perplexity, original_loss = evaluation(original_model, eval_dataloader, device)
    print(f"Original model perplexity: {original_perplexity:.2f}")

    # Evaluate compressed model
    print("Evaluating compressed model...")
    try:
        compressed_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
        compressed_perplexity, compressed_loss = evaluation(compressed_model, eval_dataloader, device)
        print(f"Compressed model perplexity: {compressed_perplexity:.2f}")
    except Exception as e:
        print(f"Error loading compressed model: {e}")
        compressed_perplexity = float("inf")

    # vLLM pipeline comparison
    print("\n" + "="*50)
    print("vLLM Pipeline Comparison")
    print("="*50)

    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In machine learning, the most important",
        "The weather today is",
    ]

    try:
        # Load models with vLLM
        original_vllm = LLM(model=model_name, max_model_len=1024)
        compressed_vllm = LLM(model=output_dir, max_model_len=1024)

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.9
        )

        # Generate with both models
        print("\nOriginal Model Outputs:")
        original_outputs = original_vllm.generate(test_prompts, sampling_params)
        for i, output in enumerate(original_outputs):
            print(f"Prompt {i+1}: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}\n")

        print("\nCompressed Model Outputs:")
        compressed_outputs = compressed_vllm.generate(test_prompts, sampling_params)
        for i, output in enumerate(compressed_outputs):
            print(f"Prompt {i+1}: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}\n")

    except Exception as e:
        print(f"Error in vLLM comparison: {e}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Original Model Perplexity: {original_perplexity:.2f}")
    print(f"Compressed Model Perplexity: {compressed_perplexity:.2f}")
    if original_perplexity != float("inf"):
        print(f"Perplexity Ratio: {compressed_perplexity/original_perplexity:.3f}")
