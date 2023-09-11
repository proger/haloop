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

    def forward(self, inputs, condtargets, input_lengths, condtarget_lengths):
        features, feature_lengths, _ = self.encoder(inputs, input_lengths, measure_entropy=True)
        loss, _ = self.recognizer(features, condtargets, feature_lengths, condtarget_lengths,
                                  measure_entropy=True, # Batching rule not implemented for aten::_chunk_grad_outputs_efficient_attention
                                  drop_labels=False)
        return loss


def compute_grad_norm(self: MiniSystem, loader):
    device = next(self.encoder.parameters()).device

    self.train() # test time dropout pls
    for dataset_indices, inputs, condtargets, input_lengths, condtarget_lengths in loader:
        inputs = inputs.to(device) # (N, T, C)
        input_lengths = input_lengths.to(device) # (N,)
        condtargets = condtargets.to(device) # (N, U)
        condtarget_lengths = condtarget_lengths.to(device) # (N,)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            norms, losses = gradient_norms(
                self,
                inputs,
                condtargets,
                input_lengths,
                condtarget_lengths.long(),
            )


        for dataset_index, norm, loss in zip(dataset_indices, norms, losses):
            print('grad_norm,loss', dataset_index.item(), norm.item(), loss.item(), sep='\t', flush=True)
        # dataset_norms = torch.stack(attempt_norms) # (A, N)
        # dataset_logits = torch.stack(attempt_logits) # (A, N)
        # dataset_grad_length = (dataset_norms.square().log() - dataset_logits).logsumexp(0) # (N,)

        # for dataset_index, grad_length in zip(dataset_indices, dataset_grad_length):
        #     print('egl score', dataset_index.item(), grad_length.item(), sep='\t', flush=True)


def make_per_sample_gradients(system, randomness='different'):
    def compute_loss(params, buffers, args):
        loss = functional_call(system, (params, buffers), args)
        return loss

    return vmap(grad_and_value(compute_loss), in_dims=(None, 0, 0), randomness=randomness)


def norm_batched(x, p=2.0, eps=1e-6):
    N = x.size(0)
    x = x.view(N, -1)
    a = x.abs().max(dim=-1, keepdim=True).values + eps
    return a.squeeze() * ((x / a).abs()**p).sum(dim=-1) ** (1./p)


def gradient_norms(
    self: MiniSystem,
    inputs: torch.Tensor, # (N, T, C)
    condtargets: torch.Tensor, # (N, U)
    input_lengths: torch.Tensor, # (N,)
    condtarget_lengths: torch.Tensor # (N,)
) -> tuple[torch.Tensor, torch.Tensor]: # (N,), (N,)
    """Compute the gradient norms for each sample in the batch independently.
    Puts system into evaluation mode.

    Returns a tuple of (gradient_norms, losses).
    """
    self.train() # test time dropout pls

    params = {k: v.detach() for k, v in self.named_parameters()}
    buffers = {k: v.detach() for k, v in self.named_buffers()}

    per_sample_gradients = make_per_sample_gradients(self)

    args = (
        inputs[:, None, :, :],
        condtargets[:, None, :],
        input_lengths[:, None],
        condtarget_lengths[:, None],
    )

    tree, losses = per_sample_gradients(params, buffers, args)
    flat_tree, spec = tree_flatten(tree)
    return norm_batched(torch.stack([norm_batched(x) for x in flat_tree]).T), losses

