import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad_and_value
from torch.utils._pytree import tree_flatten

from .recognizer import Decodable
from .transformer import CTCAttentionDecoder


class MiniSystem(nn.Module):
    def __init__(self, encoder, recognizer: Decodable):
        super().__init__()
        self.encoder = encoder
        if isinstance(recognizer, CTCAttentionDecoder):
            # ignore CTC loss as there is no batching rule for vmap
            self.recognizer = recognizer.decoder
        else:
            self.recognizer = recognizer

    def forward(self, inputs, targets, input_lengths, target_lengths):
        features, feature_lengths, _ = self.encoder(inputs, input_lengths, measure_entropy=True)
        loss, _ = self.recognizer(features, targets, feature_lengths, target_lengths, measure_entropy=True, drop_labels=False)
        return loss


def compute_grad_norm(self: MiniSystem, loader):
    device = next(self.encoder.parameters()).device

    self.eval()
    for dataset_indices, inputs, targets, input_lengths, target_lengths in loader:
        inputs = inputs.to(device) # (N, T, C)
        input_lengths = input_lengths.to(device) # (N,)
        targets = targets.to(device) # (N, U)
        target_lengths = target_lengths.to(device) # (N,)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            norms = gradient_norms(
                self,
                inputs,
                targets,
                input_lengths,
                target_lengths.long() - 1,
            )


        for dataset_index, norm in zip(dataset_indices, norms):
            print('grad_norm', dataset_index.item(), norm.item(), sep='\t', flush=True)
        # dataset_norms = torch.stack(attempt_norms) # (A, N)
        # dataset_logits = torch.stack(attempt_logits) # (A, N)
        # dataset_grad_length = (dataset_norms.square().log() - dataset_logits).logsumexp(0) # (N,)

        # for dataset_index, grad_length in zip(dataset_indices, dataset_grad_length):
        #     print('egl score', dataset_index.item(), grad_length.item(), sep='\t', flush=True)


def make_per_sample_gradients(system):
    def compute_loss(params, buffers, args):
        loss = functional_call(system, (params, buffers), args)
        return loss

    return vmap(grad_and_value(compute_loss), in_dims=(None, 0, 0))


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
) -> tuple[torch.Tensor, torch.Tensor]: # (N,), (N,)
    """Compute the gradient norms for each sample in the batch independently.
    Puts system into evaluation mode.

    Returns a tuple of (gradient_norms, losses).
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

    tree, losses = per_sample_gradients(params, buffers, args)
    flat_tree, spec = tree_flatten(tree)
    return norm_batched(torch.stack([norm_batched(x) for x in flat_tree]).T), losses

