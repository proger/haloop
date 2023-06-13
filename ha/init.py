from dataclasses import dataclass, asdict
from pathlib import Path
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lora
from .checkpoint import Checkpointer
from .attention import GPT


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    stable_embedding: bool = False
    causal: bool = True

    def state_dict(self):
        return asdict(self)

    
def load_model(ckpt_path, *, map_location):
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    if not 'vocab_size' in checkpoint['model_args']:
        # assume checkpoint for a large model

        checkpoint['model_args']['stable_embedding'] = True
        checkpoint['model_args']['vocab_size'] = 50257
        checkpoint['model_args']['bias'] = True

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'], strict=False)
    elif '_orig_mod.transformer.h.0.attn.c_attn.lora_A.weight' in checkpoint['model']:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        lora.attach_to_c_attn(model)
        model.load_state_dict(checkpoint['model'])
    else:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'])

    model.eval()
    model.to(map_location)
    model = model._orig_mod

    return model


def create_model(arch: str):
    match arch:
        case 'decoder':
            gptconf = GPTConfig()
        case 'encoder':
            gptconf = GPTConfig(block_size=128, causal=False)

    model = GPT(gptconf)
    model = torch.compile(model)
    return model


@torch.inference_mode()
def main():
    import argparse

    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='hai initializes models', formatter_class=Formatter)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('arch', choices=['decoder', 'encoder'], type=str)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = create_model(args.arch)
    print('creating a new model')
    print(model)
    print(model.config)
    Checkpointer(args.path, save_all=True)(loss=float('inf'), epoch=-1, checkpoint_fn=lambda: {
        'model': model.state_dict(),
        'model_args': model.config.state_dict()
    })


if __name__ == '__main__':
    main()
