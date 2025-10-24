#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import time
import argparse
import logging
import math
import os
import random
from pathlib import Path
from re import L
import copy

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed
)

from notebooks.check_model_quant import log_model_quantization

# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.compression.helper import recursive_getattr
from deepspeed.compression.helper import convert_conv1d_to_linear

import numpy as np
from transformers.modeling_utils import Conv1D

from torch.utils.tensorboard import SummaryWriter

from smoothquant.smoothquant.calibration import get_act_scales
from smoothquant.smoothquant.smooth import smooth_lm

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--path-to-model",
        type=str,
        help="Path to fine-tuned model or model identifier from huggingface.co/models.",
        default=None,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        '--next_reg_lam',
        type=float,
        default=0.1,
        help='Next regularization loss coeficient'
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--not_tie_wre", action="store_true", help="tie the last layer and embedding or not."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--data_folder", type=str, help="The token to use to push to the Model Hub.")
    
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local-rank",
                        type=int,
                        default=-1,
                        help="Alias for local_rank (for torch.distributed.launch compatibility)")

    parser.add_argument("--device",
                        type=int,
                        default=0,
                        help="gpu device for model ans tensors")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--use_prev_quant_layer_input", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--smooth_dataset_path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument(
        "--smooth_output_path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def save_model_checkpoint(model, output_dir, args):
    WEIGHTS_NAME = "pytorch_model.pt"
    CONFIG_NAME = 'config.json'
    output_dir = os.path.join(output_dir, 'best')    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    ### get the model to be saved
    model_to_save = model.module if hasattr(model, 'module') else model
    model_will_save = copy.deepcopy(model_to_save)
    torch.save(model_will_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    # Save args to text file
    output_args_file = 'run_script.txt'
    with open(output_args_file, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")



def main():
    args = parse_args()

    # Setup tensorboard writer
    tb_output_dir = os.path.join(args.output_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_output_dir)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if hasattr(args, 'device') and args.device is not None:
        # Single GPU case with specific device
        device = torch.device(f'cuda:{args.device}')
        torch.cuda.set_device(args.device)
    elif args.local_rank != -1:
        # Distributed case
        device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(args.local_rank)
    else:
        # Default case
        device = torch.device('cuda:0')

    print(f'Setted device: {device}')

    # Initialize distributed only if local_rank != -1
    if args.local_rank != -1:
        deepspeed.init_distributed()

    def print_rank_0(msg):
        if args.local_rank <= 0:
            print(msg)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    torch.distributed.barrier()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    if args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    #print (config)
    if args.not_tie_wre:
        config.tie_word_embeddings=False    

    if args.model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        print_rank_0("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.smooth:
        print_rank_0('start smooth operation...')
        act_scales = get_act_scales(
            model, tokenizer, args.smooth_dataset_path, args.num_samples, args.seq_len
        )

        os.makedirs(os.path.dirname(args.smooth_output_path), exist_ok=True)
        torch.save(act_scales, args.smooth_output_path)

        smooth_lm(model, act_scales, args.alpha)
        print_rank_0('smooth operation ends successfully!')
        print_rank_0(f'smooth weights saved in: {args.smooth_output_path}')

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # train_dataset = torch.load(f'{args.data_folder}/train_dataset.pt') #lm_datasets["train"]
    # eval_dataset = torch.load(f'{args.data_folder}/eval_dataset.pt') #lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     print_rank_0(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, sampler=train_sampler, batch_size=args.per_device_train_batch_size
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
 
    # Train!
    print_rank_0("***** Running training *****")
    print_rank_0(f"  Num examples = {len(train_dataset)}")
    print_rank_0(f"  Num Epochs = {args.num_train_epochs}")
    print_rank_0(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print_rank_0(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print_rank_0(f"  Total optimization steps = {args.max_train_steps}")


    num_p = sum([p.numel() for p in model.parameters()])
    print_rank_0('Number of parameters: {}'.format(num_p))

    def to_device(batch):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            # batch = tuple(t.to(device) for t in batch)
            batch = to_device(batch)
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

    def training(model, train_dataloader, eval_dataloader, num_train_epochs, args, writer):
        # Optimizer
        previous_best = None
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Set init optimizer params
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=args.num_warmup_steps,
                num_training_steps=args.max_train_steps,
            )

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            dist_init_required=True)

        # LKD Setup - Load teacher model
        print_rank_0("Initializing teacher model for LKD...")
        teacher_config = AutoConfig.from_pretrained(args.model_name_or_path)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=teacher_config,
        )
        teacher_model.to(device)
        teacher_model.eval()

        # Initialize teacher with deepspeed
        teacher_model, _, _, _ = deepspeed.initialize(
            model=teacher_model,
            args=args,
            dist_init_required=True)

        # Initial evaluation
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"Initial student perplexity: {perplexity}")

        teacher_perplexity = evaluation(teacher_model, eval_dataloader)
        print_rank_0(f"Teacher perplexity: {teacher_perplexity}")

        # LKD Training Loop
        print_rank_0("Starting Layer-wise Knowledge Distillation...")

        # Create dir for saving quantized model
        if args.output_dir is not None:
            print_rank_0('creating model save dir ...')
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)

        best_perplexity = -1

        # Set this tmp var
        prev_student_layer = None

        for l in range(model.module.config.n_layer - 1):  # GPT-2 LKD crash after last layer (that is why n_layer - 1)
            print_rank_0(f"Training layer {l}")

            # Extract student layer
            student_layer = recursive_getattr(model.module, f'transformer.h.{l}')

            student_layer_next = None
            if args.next_reg_lam > 0. and l + 2 < model.module.config.n_layer:
                student_layer_next = recursive_getattr(model.module, f'transformer.h.{l + 1}')

            # Create optimizer for this specific layer
            optimizer_param = [
                {
                    "params": [p for n, p in student_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in student_layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            # Set optimiser
            layer_optimizer = AdamW(optimizer_param, lr=args.learning_rate)
            num_training_steps = min(len(train_dataloader) * args.num_train_epochs, args.max_train_steps)

            # Set scheduler
            layer_scheduler = CosineAnnealingLR(layer_optimizer, T_max=num_training_steps)

            updated_steps = 0

            for epoch in tqdm(range(args.num_train_epochs), desc=f'Train proccessing layer {l}'):
                for step, batch in enumerate(train_dataloader):
                    batch = to_device(batch)

                    with torch.no_grad():
                        # Get teacher outputs
                        teacher_out = teacher_model(**batch, output_hidden_states=True)

                    # Extract layer inputs and outputs
                    layer_input = teacher_out.hidden_states[l]  # l-th layer input
                    teacher_o = teacher_out.hidden_states[l + 1]  # l+1-th layer output

                    # Get previous student layer if config wants
                    if prev_student_layer is not None:
                        prev_teacher_o = teacher_out.hidden_states[l - 1] # l-1-th layer output
                        layer_input = prev_student_layer(prev_teacher_o)[0]
                    
                    # Get student layer output
                    student_o = student_layer(layer_input)[0]  # GPT-2 layer forward pass

                    # Calculate MSE loss between teacher and student layer outputs
                    loss = torch.nn.functional.mse_loss(student_o, teacher_o)

                    if student_layer_next is not None:
                        teacher_o_next = teacher_out.hidden_states[l + 2]  # l-th layer output
                        student_o_next = student_layer_next(student_o)[0]
                        next_reg_lam = args.next_reg_lam
                        loss = loss + next_reg_lam * torch.nn.functional.l1_loss(student_o_next, teacher_o_next)

                    layer_optimizer.zero_grad()
                    loss.backward()
                    layer_optimizer.step()
                    layer_scheduler.step() 

                    updated_steps += 1
                    if updated_steps >= args.max_train_steps:
                        break

                    writer.add_scalar(f'training/mse_loss/layer_{l}', loss.item(), updated_steps)
                    writer.flush()

                if updated_steps >= args.max_train_steps:
                    break

            # Eval after l training
            perplexity = evaluation(model, eval_dataloader)
            print_rank_0(f"After layer {l} student eval perplexity: {perplexity}")

            writer.add_scalar('eval/perplexity', perplexity, l)
            writer.add_scalar('eval/layer', l, l)
            writer.flush()

            if args.output_dir is not None and (best_perplexity == -1 or best_perplexity > perplexity):
                best_perplexity = perplexity # Update best perplexity on train
                print_rank_0(f'saving best quantized model after tuning on layer {l}...')
                if torch.distributed.get_rank() == 0:

                    model_to_save = redundancy_clean(model, args.deepspeed_config)

                    # Log quantization part of model
                    log_model_quantization(model)

                    # Set constants
                    WEIGHTS_NAME = "quantized_model"
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

                    # torch.save(model_to_save.state_dict(), output_model_file)
                    # DeepSpeed saving tool
                    # model.save_checkpoint(output_model_file)

                    # Bert (from pypeline) saving tool
                    save_model_checkpoint(model_to_save, output_model_file, args)

                    # Calculate checkpoint file size
                    checkpoint_size = os.path.getsize(output_model_file)  # Size in bytes
                    checkpoint_size_mb = checkpoint_size / (1024 * 1024)  # Convert to MB
                    checkpoint_size_gb = checkpoint_size / (1024 * 1024 * 1024)  # Convert to GB
                    print_rank_0(f"Checkpoint file size: {checkpoint_size:,} bytes ({checkpoint_size_mb:.2f} MB / {checkpoint_size_gb:.2f} GB)")

                    tokenizer.save_vocabulary(args.output_dir)
            
            if args.use_prev_quant_layer_input:
                prev_student_layer = student_layer

        # Evaluate after LKD
        print_rank_0(f"***** Evaluating perplexity, Epoch {args.num_train_epochs}/{num_train_epochs} *****")
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"Before cleaning, Epoch at {args.num_train_epochs} with Student Perplexity: {perplexity}")


    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"Before converting the module COVN1D to linear, and before applying init_compression: {perplexity}")
    model = convert_conv1d_to_linear(model, Conv1D)
    model = init_compression(model, args.deepspeed_config)
    print_rank_0('WARNING: saving the quantized model with Linear Module instead of COV1D')

    training(model, train_dataloader, eval_dataloader, args.num_train_epochs, args, writer)

    model = redundancy_clean(model, args.deepspeed_config)
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"After cleaning with Perplexity: {perplexity}")

    quant_output_dir = args.output_dir+'/quant'
    print_rank_0(f'saving model to {quant_output_dir}')
    if not os.path.isdir(quant_output_dir):
        os.makedirs(quant_output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(quant_output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
    main()
