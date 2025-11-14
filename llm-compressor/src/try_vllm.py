import sys
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from transformers.pytorch_utils import Conv1D
from deepspeed.compression.helper import convert_conv1d_to_linear
from vllm import LLM, SamplingParams
import torch
from torch.utils.data import DataLoader, SequentialSampler
import math
import numpy as np
from transformers import default_data_collator, set_seed


def evaluation(model, eval_dataloader, eval_dataset):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        # batch = tuple(t.to(device) for t in batch)
        batch = batch.cuda()
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.cpu().item())
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(np.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    return perplexity

if __name__ == "__main__":
    set_seed(42)

    recipe = [
        # SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
    ]

    # Set params
    # model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model_name = 'openai-community/gpt2-large'
    # model_name = 'facebook/opt-350m'
    dataset_name = 'wikitext'
    dataset_subset = 'wikitext-2-raw-v1'

    # Set variables using 
    # model = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    model = convert_conv1d_to_linear(model, Conv1D)
    dataset = load_dataset(dataset_name, dataset_subset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = f'/home/buka2004/PTQ-LLM-MIPT/vllm_out/{model_name}_v2/{dataset_name}'

    oneshot(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=1024,
        num_calibration_samples=512,
    )

    # Evaluate both models block

    # Prepare evaluation data
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], truncation=True, max_length=1024)

    # tokenized_datasets = dataset.map(
    #     tokenize_function,
    #     batched=True,
    #     remove_columns=dataset["train"].column_names,
    # )

    # eval_dataset = tokenized_datasets["validation"]
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, 
    #     collate_fn=default_data_collator, 
    #     sampler=eval_sampler, 
    #     batch_size=8
    # )

    # # Evaluate original model
    # print("Evaluating original model...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # original_perplexity = evaluation(original_model, eval_dataloader, device)
    # print(f"Original model perplexity: {original_perplexity:.2f}")

    # # Evaluate compressed model
    # print("Evaluating compressed model...")
    # compressed_model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
    # compressed_perplexity = evaluation(compressed_model, eval_dataloader, device)
    # print(f"Compressed model perplexity: {compressed_perplexity:.2f}")

    # # vLLM pipeline comparison
    # print("\n" + "="*50)
    # print("vLLM Pipeline Comparison")
    # print("="*50)

    # # Test prompts
    # test_prompts = [
    #     "The future of artificial intelligence is",
    #     "In machine learning, the most important",
    #     "The weather today is",
    # ]

    # # Load models with vLLM
    # original_vllm = LLM(model=model_name, max_model_len=1024)
    # compressed_vllm = LLM(model=output_dir, max_model_len=1024)

    # sampling_params = SamplingParams(
    #     temperature=0.7,
    #     max_tokens=50,
    #     top_p=0.9
    # )

    # # Generate with both models
    # print("\nOriginal Model Outputs:")
    # original_outputs = original_vllm.generate(test_prompts, sampling_params)
    # for i, output in enumerate(original_outputs):
    #     print(f"Prompt {i+1}: {output.prompt}")
    #     print(f"Generated: {output.outputs[0].text}\n")

    # print("\nCompressed Model Outputs:")
    # compressed_outputs = compressed_vllm.generate(test_prompts, sampling_params)
    # for i, output in enumerate(compressed_outputs):
    #     print(f"Prompt {i+1}: {output.prompt}")
    #     print(f"Generated: {output.outputs[0].text}\n")

    # # Print summary
    # print("\n" + "="*50)
    # print("SUMMARY")
    # print("="*50)
    # print(f"Original Model Perplexity: {original_perplexity:.2f}")
    # print(f"Compressed Model Perplexity: {compressed_perplexity:.2f}")
    # print(f"Perplexity Ratio: {compressed_perplexity/original_perplexity:.3f}")