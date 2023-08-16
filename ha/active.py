
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


def norm_batched(x, p=2.0, eps=1e-6):
    N = x.size(0)
    x = x.view(N, -1)
    a = x.abs().max(dim=-1, keepdim=True).values + eps
    return a.squeeze() * ((x / a).abs()**p).sum(dim=-1) ** (1./p)


def gradient_norms(
    system: MiniSystem,
    inputs: torch.Tensor, # (N, T, C)
    targets: torch.Tensor, # (N, U)
    input_lengths: torch.Tensor, # (N,)
    target_lengths: torch.Tensor # (N,)
) -> torch.Tensor: # (N,)
    """Compute the gradient norms for each sample in the batch independently.
    Puts system into evaluation mode.
    """
    system.eval()

    params = {k: v.detach() for k, v in system.named_parameters()}
    buffers = {k: v.detach() for k, v in system.named_buffers()}

    per_sample_gradients = make_per_sample_gradients(system)

    args = (
        inputs[:, None, :, :],
        targets[:, None, :],
        input_lengths[:, None],
        target_lengths[:, None],
    )

    tree = per_sample_gradients(params, buffers, args)
    flat_tree, spec = tree_flatten(tree)
    return norm_batched(torch.stack([norm_batched(x) for x in flat_tree]).T)

