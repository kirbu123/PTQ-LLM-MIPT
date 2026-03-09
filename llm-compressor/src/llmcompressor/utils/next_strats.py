import torch
import torch.nn as nn
from llmcompressor.modifiers.utils.kernels import apply_conv
from abc import abstractmethod

__all__ = ["BasicStrat", "AllLinears"]

def BasicStrat(postfix: str):
  return [postfix]

def AllLinears(postfix: str):
  return list(set([postfix, 'out_proj', 'fc1', 'fc2']))

def IgnoreNotOutProj(postfix: str):
  if postfix == 'out_proj':
    return [postfix]
  else:
    return []


NEXT_STRATS_DICT = {
    'BasicStrat': BasicStrat,
    'AllLinears': AllLinears
}
