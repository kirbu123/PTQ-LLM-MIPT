import time
import argparse
import logging
import math
import os
import copy
import random
from pathlib import Path
from re import L

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

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
    set_seed,
)
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
from smoothquant.smoothquant.fake_quant import quantize_model


def save_checkpoint(model, output_dir):
    WEIGHTS_NAME = "pytorch_model.bin"
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


if __name__ == "__main__":
  # Configuration
  MODEL = 'openai-community/gpt2-large'
  CONFIG = '/home/buka2004/PTQ-LLM-MIPT/DeepSpeedExamples/compression/gpt2/config/ds_config_W8A8_Qgroup64_fp32.json'
  SAVE_PATH = '/home/buka2004/PTQ-LLM-MIPT/W8A8_quantization_lkd_saving'

  # Create save directory
  os.makedirs(SAVE_PATH, exist_ok=True)

  with open(CONFIG, 'r') as file:
    ds_config = json.load(file)

  model_config = AutoConfig.from_pretrained(MODEL)
  model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    config=model_config
  )
  # model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
  ds_engine, _, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
  # lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(ds_engine.module.state_dict())
  # ds_engine.module.save_pretrained("lean_after", state_dict=lean_state_dict)

  quantized_model = convert_conv1d_to_linear(ds_engine.module, Conv1D)
  # quantized_model = quantize_model(quantized_model)

  # Save using torch directly in DeepSpeed format
  # checkpoint = {
  #   'module': quantized_model.state_dict(),
  #   'optimizer': {},
  #   'lr_scheduler': None,
  #   'client_state': {}
  # }
  # torch.save(checkpoint, f"{SAVE_PATH}/quantized_model.pt")

  save_checkpoint(quantized_model, SAVE_PATH)

  print(f"Quantized model saved to: {SAVE_PATH}")
