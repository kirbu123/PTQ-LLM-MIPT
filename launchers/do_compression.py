import argparse
import os
import sys
sys.path.append('./llm-compressor/src')
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier, SmoothQuantRegModifier
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from transformers.pytorch_utils import Conv1D
from transformers import set_seed
import subprocess
from deepspeed.compression.helper import convert_conv1d_to_linear
import torch

BYTES_PRECISION_DICT = {
    torch.float32: 4,
    torch.float16: 2,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1
}

def estimate_model_params(model):
    """Estimate model memory usage based on parameters and precision"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_byte_params = sum(p.numel() * BYTES_PRECISION_DICT[p.dtype] for p in model.parameters())

    # Calculate memory based on precision
    model_memory_gb = total_byte_params / (1024**3)
    
    # For inference, we typically need model weights + some overhead
    inference_memory_gb = model_memory_gb * 1.2  # 20% overhead for activations

    param_info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_memory_gb': model_memory_gb,
        'inference_memory_gb': inference_memory_gb
    }
    
    return param_info

def evaluate_with_lm_eval(
        model_path, model_name, teacher_model, student_model, tasks="wikitext",
        num_fewshot=0, limit=500, device="cuda:0",
        eval_log_name='evaluation_logs', eval_res_name='evaluation_results'
    ):
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

    print("\n" + "="*50)
    print("Starting lm_eval evaluation")
    print("="*50)

    # Log teacher model params

    teacher_param_info = estimate_model_params(teacher_model)
    compressed_param_info = estimate_model_params(student_model)

    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"

    cmd_teacher = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name}",
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--batch_size", "1",
    ]

    cmd_student = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--batch_size", "1",
        "--output_path",  f"{model_path}"
    ]

    print(f"Running lm_eval with commands:")
    print("TEACHER:")
    print(" ".join(cmd_teacher))
    print("STUDENT:")
    print(" ".join(cmd_student))
    print(f"Using device: {device}")

    try:
        # Run the command
        result_teacher = subprocess.run(cmd_teacher, env=env, capture_output=True, text=True)
        result_student = subprocess.run(cmd_student, env=env, capture_output=True, text=True)
        
        # Save results to file
        eval_log_file = os.path.join(model_path, f"{eval_log_name}.log")
        eval_res_file = os.path.join(model_path, f"{eval_res_name}.txt")

        with open(eval_res_file, "w") as f:
            f.write("MODELS SIZE RESULTS\n")
            f.write("=" * 50 + "\n")
            
            # Teacher model parameters
            f.write("TEACHER MODEL:\n")
            f.write(f"  Total parameters: {teacher_param_info['total_params']:,}\n")
            f.write(f"  Trainable parameters: {teacher_param_info['trainable_params']:,}\n")
            f.write(f"  Model memory: {teacher_param_info['model_memory_gb']:.2f} GB\n")
            f.write(f"  Estimated inference memory: {teacher_param_info['inference_memory_gb']:.2f} GB\n")
            f.write("\n")
            
            # Student (compressed) model parameters
            f.write("STUDENT MODEL (COMPRESSED):\n")
            f.write(f"  Total parameters: {compressed_param_info['total_params']:,}\n")
            f.write(f"  Trainable parameters: {compressed_param_info['trainable_params']:,}\n")
            f.write(f"  Model memory: {compressed_param_info['model_memory_gb']:.2f} GB\n")
            f.write(f"  Estimated inference memory: {compressed_param_info['inference_memory_gb']:.2f} GB\n")
            f.write("\n")
            
            # Compression ratio
            compression_ratio = teacher_param_info['model_memory_gb'] / compressed_param_info['model_memory_gb']
            memory_reduction = (1 - compressed_param_info['model_memory_gb'] / teacher_param_info['model_memory_gb']) * 100
            f.write("COMPRESSION RESULTS:\n")
            f.write(f"  Compression ratio: {compression_ratio:.2f}x\n")
            f.write(f"  Memory reduction: {memory_reduction:.2f}%\n")

            f.write("=" * 50 + "\n\n")

            f.write("LM-EVAL EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model name: {model_name}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Tasks: {tasks}\n")
            f.write(f"Few-shot: {num_fewshot}\n")
            f.write(f"Limit: {limit}\n")
            f.write(f"Device: {device}\n")
            f.write("=" * 50 + "\n\n")

            if result_teacher.stdout:
                f.write("TEACHER EVALUATION RESULTS:\n")
                f.write(result_teacher.stdout)
                f.write("\n")

            f.write("=" * 50 + "\n")

            if result_student.stdout:
                f.write("STUDENT EVALUATION RESULTS:\n")
                f.write(result_student.stdout)
                f.write("\n")
            
            f.write("=" * 50 + "\n")

        with open(eval_log_file, "w") as f:
            f.write("=" * 50 + "\n")

            if result_teacher.stderr:
                f.write("TEACHER STDERR:\n")
                f.write(result_teacher.stderr)
                f.write("\n")

            f.write(f"\nReturn code: {result_teacher.returncode}\n")

            f.write("=" * 50 + "\n")

            if result_student.stderr:
                f.write("STUDENT STDERR:\n")
                f.write(result_student.stderr)
                f.write("\n")

            f.write(f"\nReturn code: {result_student.returncode}\n")

            f.write("=" * 50 + "\n")

        print(f"✓ Evaluation results saved to: {eval_log_file} and {eval_res_file}")

        # Also print results to console
        if result_teacher.stdout:
            print("\nTEACHER EVALUATION RESULTS:")
            print("=" * 50)
            print(result_teacher.stdout)

        if result_student.stdout:
            print("\nSTUDENT EVALUATION RESULTS:")
            print("=" * 50)
            print(result_student.stdout)


        return result_teacher.returncode == 0 and result_student.returncode == 0

    except Exception as e:
        print(f"Error running lm_eval: {e}")
        return False


def quantize_model_by_oneshot(
        model_base_path, dataset_base_path,
        model_name, dataset_name, dataset_subset, output_dir,
        scheme="W8A8", targets="Linear", ignore=["lm_head"], next_reg_lam=0., next_loss_lam=0., kernel_mode='default',
        max_seq_length=1024, num_calibration_samples=512, smoothing_strength=0.5,
        hes_reg_lam=0.1, gptq=True, smoothquant=False, smoothquantreg=True, lam_optimize=False, do_hessian_plot=False, lam_lr=3e-4, k_next=1, opt_steps_num=10,
        lam_loss_name='HessianLossNormed', next_strat_name='BasicStrat',
        seed=42
    ):

    output_dir = os.path.join(output_dir, model_name, dataset_name, f'smoothing_strength={smoothing_strength}:next_reg_lam={next_reg_lam}:next_loss_lam={next_loss_lam}:lam_lr={lam_lr}:k_next={k_next}:opt_steps_num={opt_steps_num}:kernel_mode={kernel_mode}:hes_reg_lam={hes_reg_lam}:seed={seed}')

    if smoothquant and smoothquantreg:
        ValueError('Evailable to use only one smooth method, picked two')

    recipe = []
    if smoothquant:
        recipe = recipe + [SmoothQuantModifier(smoothing_strength=smoothing_strength)]
    if smoothquantreg:
        recipe = recipe + [SmoothQuantRegModifier(smoothing_strength=smoothing_strength, hes_reg_lam=hes_reg_lam)]
    if gptq:
        recipe = recipe + [GPTQModifier(
            scheme=scheme,
            targets=targets,
            ignore=ignore,
            next_reg_lam=next_reg_lam,
            next_loss_lam=next_loss_lam,
            kernel_mode=kernel_mode,
            log_dir=output_dir,
            lam_optimize=lam_optimize,
            do_hessian_plot=do_hessian_plot,
            lam_lr=lam_lr,
            k_next=k_next,
            opt_steps_num=opt_steps_num,
            lam_loss_name=lam_loss_name,
            next_strat_name=next_strat_name,
        )]

    # Set dataset 
    if dataset_base_path is not None:
        dataset_path = os.path.join(dataset_base_path, dataset_name, dataset_subset)
        dataset = load_dataset(dataset_path)
    else:
        dataset = load_dataset(dataset_name, dataset_subset)

    # Set model
    if model_base_path is not None:
        model_path = os.path.join(model_base_path, model_name)
    else:
        model_path = model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_path
    )

    if 'gpt' in model_name.lower():
        model = convert_conv1d_to_linear(model, Conv1D)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    oneshot_model = oneshot(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

    return oneshot_model, model, model_name, output_dir, dataset, tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize a model using GPTQ oneshot compression')
    
    # Device param
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='openai-community/gpt2-large',
                       help='HuggingFace model name or path')

    parser.add_argument("--dataset_base_path", default=None, type=str, help='base dir path for dataset')
    parser.add_argument("--model_base_path", default=None, type=str, help='base dir path for model')
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                       help='Dataset name for calibration')
    parser.add_argument('--dataset_subset', type=str, default='wikitext-2-raw-v1',
                       help='Dataset subset for calibration')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for quantized model (optional)')

    # Param optimize
    parser.add_argument('--lam_optimize', action='store_true',
                   help='Enable lam param optimize')
    
    # Hessian plot
    parser.add_argument('--do_hessian_plot', action='store_true',
                   help='Enable hessian plot')

    # GPTQ parameters
    parser.add_argument('--gptq', action='store_true',
                   help='Enable GPTQ quantization')
    parser.add_argument('--scheme', type=str, default='W8A8',
                       choices=['W8A8', 'W4A8', 'W4A16', 'W2A16'],
                       help='Quantization scheme')
    parser.add_argument('--targets', type=str, default='Linear',
                       help='Target modules to quantize (comma-separated)')
    parser.add_argument('--ignore', type=list, default=['lm_head'],
                       help='Modules to ignore during quantization (comma-separated)')
    parser.add_argument('--next_reg_lam', type=float, default=0.0,
                       help='Regularization parameter for next layer influence')
    parser.add_argument('--next_loss_lam', type=float, default=0.0,
                       help='Regularization parameter for next layer loss influence')
    parser.add_argument('--kernel_mode', type=str, default='default',
                       help='Kernel type of the conv, using for local attention')
    parser.add_argument('--num_calibration_samples', type=int, default=512,
                       help='Number of calibration samples')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                       help='Maximum sequence length')

    # SmoothQuant && SmoothQuantReg
    parser.add_argument('--smoothquantreg', action='store_true',
                   help='Enable SmoothQuant quantization')
    parser.add_argument('--smoothquant', action='store_true',
                   help='Enable SmoothQuant quantization')
    parser.add_argument('--smoothing_strength', type=float, default=0.5,
                       help='Regularization alpha parameter for smoothquant method')
    parser.add_argument('--hes_reg_lam', type=float, default=0.1,
                       help='Regularization lam parameter for smoothquant weight hesian reg')
    parser.add_argument('--lam_lr', type=float, default=3e-4,
                       help='Regularization lam parameter optimization lr')
    parser.add_argument('--k_next', type=int, default=1,
                       help='Regularization lam parameter optimization lr')
    parser.add_argument('--opt_steps_num', type=int, default=10,
                       help='Regularization lam parameter optimization lr')
    parser.add_argument('--lam_loss_name', type=str, default='HessianLossNormed',
                       help='Regularization lam parameter optimization loss name')
    parser.add_argument('--next_strat_name', type=str, default='BasicStrat',
                    help='Next strategy selection name for next strat')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    set_seed(args.seed)

    # Run llm-compressor quantization
    oneshot_model, teacher_model, model_name, model_output_path, dataset, tokenizer = quantize_model_by_oneshot(
        model_base_path=args.model_base_path,
        dataset_base_path=args.dataset_base_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        output_dir=args.output_dir,
        scheme=args.scheme,
        targets=args.targets,
        ignore=args.ignore,
        next_reg_lam=args.next_reg_lam,
        next_loss_lam=args.next_loss_lam,
        kernel_mode=args.kernel_mode,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        gptq=args.gptq,
        smoothquant=args.smoothquant,
        smoothquantreg=args.smoothquantreg,
        hes_reg_lam=args.hes_reg_lam,
        smoothing_strength=args.smoothing_strength,
        lam_optimize=args.lam_optimize,
        do_hessian_plot=args.do_hessian_plot,
        lam_lr=args.lam_lr,
        k_next=args.k_next,
        opt_steps_num=args.opt_steps_num,
        lam_loss_name=args.lam_loss_name,
        next_strat_name=args.next_strat_name,
        seed=args.seed
    )

    # Run lm_eval evaluation
    success = evaluate_with_lm_eval(
        model_path=model_output_path,
        model_name=model_name,
        teacher_model=teacher_model,
        student_model=oneshot_model,
        tasks="wikitext,hellaswag,piqa,arc_easy,lambada_openai",
        num_fewshot=0,
        limit=500,
        device=args.device
    )

    if success:
        print("✓ lm_eval evaluation completed successfully")
    else:
        print("✗ lm_eval evaluation failed")
