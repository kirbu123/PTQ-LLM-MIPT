import torch
import torch.nn as nn
from llmcompressor.modifiers.utils.kernels import apply_conv
from abc import abstractmethod

__all__ = ["BasicStrat", "AllLinears"]

def BasicStrat(postfix: str):
  return [postfix]

linear_layers = [
  'out_proj', 'fc1', 'fc2', 'o_proj', 'gate_proj', 'c_fc', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'
]
def AllLinears(postfix: str):
  global linear_layers
  if postfix in linear_layers:
    return list(set([postfix] + linear_layers))
  else:
    return []

def IgnoreNotOutProj(postfix: str):
  if postfix == 'out_proj' or postfix == 'o_proj':
    return [postfix]
  else:
    return []


NEXT_STRATS_DICT = {
    'BasicStrat': BasicStrat,
    'AllLinears': AllLinears,
    'IgnoreNotOutProj': IgnoreNotOutProj
}
