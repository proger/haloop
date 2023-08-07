import torch.nn as nn



def register_activation_stat_hooks(m):
    """
    adds hooks to all activations and keeps track of their means, stds, and how many are near zero
    """
    def hook(m, i, o):
        if isinstance(o, tuple):
            o = o[0]
        m.act_mean = o.mean()
        m.act_std = o.std()
        m.act_near_zero = (o.abs() < 1e-3).float().mean()
    handles = [m.register_forward_hook(hook)]
    for name, child in m.named_modules():
        handles.append(child.register_forward_hook(hook))
    return handles


def print_activation_stat_hooks(m):
    """
    prints the activation stats of a module
    """
    for name, child in m.named_modules():
        if hasattr(child, 'act_mean'):
            print(f'{name}\t{child.act_mean:.3f} {child.act_std:.3f} {child.act_near_zero:.3f}')