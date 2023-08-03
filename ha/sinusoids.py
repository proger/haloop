import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoids_like(x, base=10000):
    _, T, C = x.shape
    t = torch.arange(0, T, dtype=x.dtype, device=x.device)[:, None]
    exp = -torch.arange(0, C, 2, dtype=x.dtype, device=x.device) / C
    even = torch.sin((base**exp) * t)
    odd = torch.cos((base**exp) * t)
    return torch.stack([even, odd], dim=-1).flatten(-2, -1)


class SyntheticAlignments(torch.utils.data.Dataset):
    def __init__(self, examples_per_bin=3000, min=100, max=16000, step=100, vocab_size=512, dim=80, seed_offset=0):
        super().__init__()
        self.min = min
        self.max = max
        self.step = step
        self.bins = (self.max - self.min) // self.step + 1
        self.examples_per_bin = examples_per_bin
        self.sinusoids = sinusoids_like(torch.randn(1, vocab_size, dim))
        self.vocab_size = vocab_size
        self.seed_offset = seed_offset

    def __len__(self):
        return self.bins * self.examples_per_bin

    def __getitem__(self, index):
        time_steps = self.min + (index % self.bins) * self.step
        generator = torch.Generator().manual_seed(self.seed_offset + index)

        repeats = torch.randint(5, 20, (time_steps,), generator=generator)
        repeats = torch.poisson(repeats.float(), generator=generator).int()
        targets = torch.randint(1, self.vocab_size, (time_steps,), generator=generator)
        total_frames = repeats.cumsum(0)

        alignments = torch.cat([torch.tensor([t]*r) for t, r, T in zip(targets, repeats, total_frames) if T < time_steps]).int()
        targets = torch.LongTensor([t for t, T in zip(targets, total_frames) if T < time_steps])
        inputs = self.sinusoids[alignments, :]
        target_str = " ".join(map(str, targets.tolist()))
        return index, inputs, target_str


if __name__ == '__main__':
    V = 512
    sinusoids = sinusoids_like(torch.randn(1, V, 80))

    _, seq, frames = SyntheticAlignments(examples_per_bin=1)[5]

    import matplotlib.pyplot as plt
    fig, (top, bot) = plt.subplots(2, 1, sharex=True, sharey=True)
    top.matshow(sinusoids.T, cmap='Blues', aspect=1)
    bot.matshow(seq.T, cmap='Greens', aspect=1)
    top.set_axis_off()
    bot.set_axis_off()
    top.set_anchor('W')
    bot.set_anchor('W')
    top.text(-2.0,-2.0, f'{V} positions  ', size=5)
    bot.text(-2.0,-2.0, f'example encoder input sequence', size=5)
    plt.savefig('sinusoids.png', dpi=300, bbox_inches='tight')

