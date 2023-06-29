import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LayerNorm, Block


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    scales = torch.arange(channels // 2) / (channels // 2 - 1)
    inv_timescales = torch.exp(-math.log(max_timescale) * scales)
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.conv_pre = nn.Conv1d(config.d_input, config.n_embd, kernel_size=3, stride=1, padding=1)
        self.conv_subsample = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, stride=2, padding=1)

        self.transformer = nn.ModuleDict(dict(
            #wpe = nn.Embedding(config.block_size, config.n_embd),

            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        #self.transformer.wpe.weight.data = sinusoids(config.block_size, config.n_embd)
        #self.transformer.wpe.requires_grad_(False)

    def subsampled_lengths(self, input_lengths):
        # https://github.com/vdumoulin/conv_arithmetic
        p, k, s = self.conv_subsample.padding[0], self.conv_subsample.kernel_size[0], self.conv_subsample.stride[0]
        o = input_lengths + 2 * p - k
        o = torch.floor(o / s + 1)
        return o.int()

    def forward(self, x, measure_entropy=False):
        x = x.mT
        x = F.gelu(self.conv_pre(x))
        x = F.gelu(self.conv_subsample(x))
        x = x.mT

        _, t, c = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, t)
        #pe = self.transformer.wpe(pos)
        #x = self.transformer.drop(x + pe) # shape (b, t, c)
        x = self.transformer.drop(x) # shape (b, t, c)

        for i, block in enumerate(self.transformer.h):
            x, _att_entropy, _present = block(x, past=None, measure_entropy=measure_entropy)
        x = self.transformer.ln_f(x)

        return x


if __name__ == '__main__':
    from ha.init import AudioEncoderConfig
    config = AudioEncoderConfig()
    encoder = AudioEncoder(config)
    print(encoder(torch.randn(1, config.block_size, config.d_input)))
