import argparse
import os
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from transformers.pytorch_utils import Conv1D
from transformers import set_seed
import subprocess
from deepspeed.compression.helper import convert_conv1d_to_linear


def evaluate_with_lm_eval(model_path, tasks="wikitext", num_fewshot=0, limit=500, device="cuda:0"):
    """
    Evaluate model using lm-evaluation-harness and save results to evaluation.txt
    
    Args:
        model_path: Path to the model
        tasks: Comma-separated tasks
        num_fewshot: Number of few-shot examples
        limit: Number of examples to evaluate
        device: CUDA device to use
    
    Returns:
        bool: Success status
    """
    
    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
    
    # Build the command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--batch_size", "1"
    ]
    
    print(f"Running lm_eval with command:")
    print(" ".join(cmd))
    print(f"Using device: {device}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Save results to file
        eval_file = os.path.join(model_path, "evaluation.txt")
        with open(eval_file, "w") as f:
            f.write("LM-EVAL EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Tasks: {tasks}\n")
            f.write(f"Few-shot: {num_fewshot}\n")
            f.write(f"Limit: {limit}\n")
            f.write(f"Device: {device}\n")
            f.write("=" * 50 + "\n\n")
            
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n")
            
            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
                f.write("\n")
            
            f.write(f"\nReturn code: {result.returncode}\n")
        
        print(f"✓ Evaluation results saved to: {eval_file}")
        
        # Also print results to console
        if result.stdout:
            print("\nEVALUATION RESULTS:")
            print("=" * 50)
            print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running lm_eval: {e}")
        return False

def quantize_model_by_oneshot(
        model_name, dataset_name, dataset_subset, output_dir,
        scheme="W8A8", targets="Linear", ignore=["lm_head"], next_reg_lam=0.,
        max_seq_length=1024, num_calibration_samples=512):

    recipe = [
        # SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(scheme=scheme, targets=targets, ignore=ignore, next_reg_lam=next_reg_lam),
    ]

    # Set variables using 
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    if 'gpt' in model_name.lower():
        model = convert_conv1d_to_linear(model, Conv1D)

    dataset = load_dataset(dataset_name, dataset_subset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = os.path.join(output_dir, model_name, dataset_name, f'next_reg_lam={next_reg_lam}')

    oneshot_model = oneshot(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

    return oneshot_model, output_dir, dataset, tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize a model using GPTQ oneshot compression')
    
    # Device param
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='openai-community/gpt2-large',
                       help='HuggingFace model name or path')
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                       help='Dataset name for calibration')
    parser.add_argument('--dataset_subset', type=str, default='wikitext-2-raw-v1',
                       help='Dataset subset for calibration')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for quantized model (optional)')
    
    # GPTQ parameters
    parser.add_argument('--scheme', type=str, default='W8A8',
                       choices=['W8A8', 'W4A8', 'W4A16', 'W2A16'],
                       help='Quantization scheme')
    parser.add_argument('--targets', type=str, default='Linear',
                       help='Target modules to quantize (comma-separated)')
    parser.add_argument('--ignore', type=list, default=['lm_head'],
                       help='Modules to ignore during quantization (comma-separated)')
    parser.add_argument('--next_reg_lam', type=float, default=0.2,
                       help='Regularization parameter for next layer influence')
    parser.add_argument('--num_calibration_samples', type=int, default=512,
                       help='Number of calibration samples')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                       help='Maximum sequence length')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    set_seed(args.seed)

    oneshot_model, model_output_path, dataset, tokenizer = quantize_model_by_oneshot(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        output_dir=args.output_dir,
        scheme=args.scheme,
        targets=args.targets,
        ignore=args.ignore,
        next_reg_lam=args.next_reg_lam,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )

    # Run lm_eval evaluation
    print("\n" + "="*50)
    print("Starting lm_eval evaluation")
    print("="*50)
    
    success = evaluate_with_lm_eval(
        model_path=model_output_path,
        tasks="wikitext",
        num_fewshot=0,
        limit=500,
        device=args.device
    )

    if success:
        print("✓ lm_eval evaluation completed successfully")
    else:
        print("✗ lm_eval evaluation failed")
