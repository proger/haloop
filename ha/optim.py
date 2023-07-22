#
# This optimizer code is based on https://github.com/karpathy/nanoGPT/blob/master/model.py
# 

import inspect
import math
import torch

from .attention import LayerNorm


class LR:
    def __init__(self, args):
        self.args = args

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--lr', type=float, default=3e-4, help='AdamW learning rate')
        parser.add_argument('--lr_schedule', type=str, choices=['const', 'cosine'], default='cosine', help='Learning rate schedule')
        parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warm-up steps')
        parser.add_argument('--lr_decay_iters', type=int, default=200000, help='Number of steps for learning rate decay')
        parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        parser.add_argument('--beta1', type=float, default=0.9, help='Decay factor for first gradient moment')
        parser.add_argument('--beta2', type=float, default=0.99, help='Decay factor for second gradient moment')

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        args = self.args

        if args.lr_schedule == 'cosine':
            # 1) linear warmup for warmup_iters steps
            if it < args.warmup_iters:
                return args.lr * it / args.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > args.lr_decay_iters:
                return args.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return args.min_lr + coeff * (args.lr - args.min_lr)
        else:
            return args.lr

    def apply_lr_(self, optimizer, step):
        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def configure_optimizers(self, args, device_type='cuda', decay_lm_head=True):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    if decay_lm_head:
        # for decoder-only models with tied outputs:
        #
        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2), **extra_args)

    return optimizer

