#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
#
# Original repo root: https://github.com/microsoft/LoRA
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=kwargs.get('bias', False))
            self.lora_B = nn.Linear(r, out_features, bias=kwargs.get('bias', False))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [True, False, True],
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2,
                bias=False,
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad(delta_w * self.scaling)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0), 
                        self.lora_B.data.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad(delta_w * self.scaling)
                self.merged = True        

    def forward(self, x: torch.Tensor):
        if self.merged:
            return F.linear(x, self.weight, bias=self.bias)
        else:
            result = F.linear(x, self.weight, bias=self.bias)
            if self.r > 0:
                after_A = self.lora_A(self.lora_dropout(x))
                after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


def attach_to_c_attn(self, r=4, lora_alpha=32, lora_dropout=0.1):
    key_list = [key for key, _ in self.named_modules()]
    for key in key_list:
        if any(key.endswith(target_key) for target_key in ["c_attn"]):
            parent, target, target_name = _get_submodules(self, key)
            bias = target.bias is not None
            new_module = Linear(target.in_features, target.out_features, bias=bias,
                                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            _replace_module(parent, target_name, new_module, target)


def _get_submodules(self, key):
    parent = self.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = self.get_submodule(key)
    return parent, target, target_name


def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if old_module.bias is not None:
        new_module.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)
