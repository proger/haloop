import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv1d(nn.Module):
    "Depthwise separable convolution"
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size,
        self.stride = stride,
        self.padding = padding,
        self.dilation = dilation,
        self.bias = bias,
        self.depthwise = nn.Conv1d(in_channels, in_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvEncoder(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int, strides: tuple[int, ...], kernel_size: int = 3):
        super().__init__()

        conv = [nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=strides[0], padding=1)]
        for stride in strides[1:-1]:
            conv.append(DWConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1))
        conv.append(DWConv1d(hidden_dim, output_dim, kernel_size=kernel_size, stride=strides[-1], padding=1))
        self.conv = nn.ModuleList(conv)

    def subsampled_lengths(self, input_lengths):
        # https://github.com/vdumoulin/conv_arithmetic
        o = input_lengths
        for conv in self.conv:
            p, k, s = conv.padding[0], conv.kernel_size[0], conv.stride[0]
            o = o + 2 * p - k
            o = torch.floor(o / s + 1)
        return o.int()

    def forward(self, x, input_lengths):
        for conv in self.conv:
            x = F.gelu(conv(x))
        return x, self.subsampled_lengths(input_lengths)

