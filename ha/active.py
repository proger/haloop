
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_flatten, tree_map

from .recognizer import Decodable
from .transformer import CTCAttentionDecoder


class MiniSystem(nn.Module):
    def __init__(self, encoder, decoder: Decodable):
        super().__init__()
        self.encoder = encoder
        if isinstance(decoder, CTCAttentionDecoder):
            # ignore CTC loss as there is no batching rule for vmap
            self.decoder = decoder.decoder
        else:
            self.decoder = decoder

    def forward(self, x, y, xl, yl):
        x, xl, _ = self.encoder(x, xl, measure_entropy=True)
        loss, _ = self.decoder(x, y, xl, yl, measure_entropy=True)
        return loss


def make_per_sample_gradients(system):
    def compute_loss(params, buffers, args):
        loss = functional_call(system, (params, buffers), args)
        return loss

    return vmap(grad(compute_loss), in_dims=(None, 0, 0))


def gradient_norms(
    system: MiniSystem,
    features: torch.Tensor, # (N, T, C)
    targets_kbest: torch.Tensor, # (K, N, U)
    input_lengths: torch.Tensor, # (N,)
    target_kbest_lengths: torch.Tensor # (K, N)
):
    system.eval()

    K, N, U = targets_kbest.shape

    params = {k: v.detach() for k, v in system.named_parameters()}
    buffers = {k: v.detach() for k, v in system.named_buffers()}

    per_sample_gradients = make_per_sample_gradients(system)

    args = (
        features[:, None, :, :].repeat_interleave(K, 0),   # (K*N, 1, T, C)
        targets_kbest[:, :, None, :].view(-1, 1, U),       # (K*N, 1, U)
        input_lengths[:, None].repeat_interleave(K, 0),    # (K*N, 1)
        target_kbest_lengths[:, :, None].view(-1, 1)       # (K*N, 1)
    )

    tree = per_sample_gradients(params, buffers, args)
    flat_tree, spec = tree_flatten(tree)
    return torch.norm(torch.stack([torch.norm(x.sum(0), 2.0) for x in flat_tree]), 2.0)

