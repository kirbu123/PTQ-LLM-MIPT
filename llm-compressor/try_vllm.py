import sys
from src.llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from src.llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from src.llmcompressor import oneshot
from transformers.pytorch_utils import Conv1D
from deepspeed.compression.helper import convert_conv1d_to_linear

if __name__ == "__main__":
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

    output_dir = f'/home/buka2004/PTQ-LLM-MIPT/vllm_out/{model_name}/{dataset_name}'

    oneshot(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=1024,
        num_calibration_samples=512,
    )
