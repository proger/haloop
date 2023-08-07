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
    def __init__(self, examples_per_bin=3000, min=10, max=16000, step=10, vocab_size=512, dim=80, seed_offset=0):
        super().__init__()
        self.min = min
        self.max = max
        self.step = step
        self.bins = (self.max - self.min) // self.step + 1
        self.examples_per_bin = examples_per_bin
        self.vocab_size = vocab_size
        self.seed_offset = seed_offset
        self.sinusoids = sinusoids_like(torch.zeros(1, vocab_size, dim))

    def __len__(self):
        return self.bins * self.examples_per_bin

    def __getitem__(self, index):
        time_steps = self.min + (index % self.bins) * self.step
        generator = torch.Generator().manual_seed(self.seed_offset + index)

        t = 0
        targets, durations = [], []
        while t < time_steps:
            duration = torch.randint(10, 20, (1,), generator=generator).item()
            durations.append(duration)
            # 0 is pad, 1 is ???, 2 is stx, 3 is etx
            target = torch.randint(4, self.vocab_size, (1,), generator=generator).item()
            targets.append(target)
            t += duration

        alignments = torch.cat([torch.LongTensor([t]*r) for t, r in zip(targets, durations)])
        inputs = self.sinusoids[alignments, :]
        return index, inputs, " ".join(map(str, targets))


if __name__ == '__main__':
    V = 512
    torch.manual_seed(2)
    sinusoids = sinusoids_like(torch.zeros(1, V, 80))

    alignments = SyntheticAlignments(examples_per_bin=1000000, max=100)
    #alignments = SyntheticAlignments(examples_per_bin=20000, max=3000)

    import matplotlib.pyplot as plt

    if True:
        bases = [10, 50, 100, 200, 500, 700, 1000]
        B = len(bases)
        fig, axs = plt.subplots(B, 1, sharex=True, sharey=True, figsize=(8, B))

        for ax, base in zip(axs, bases):
            sinusoids = sinusoids_like(torch.zeros(1, V, 80), base=base)

            ax.matshow(sinusoids.T, cmap='Blues', aspect=1)
            ax.set_axis_off()
            ax.set_anchor('W')
            ax.text(-2.0,-2.0, f'base: {base}', size=5)
        plt.savefig('bases.png', dpi=300, bbox_inches='tight')
        print('plotting different sinusoid bases into bases.png')
    else:
        N = 16
        indices = torch.randint(0, len(alignments), (N,)).tolist()

        fig, axs = plt.subplots(N, 1, sharex=True, sharey=True, figsize=(8, N))
        for ax, index in zip(axs, indices):
            _, seq, frames = alignments[index]
            ax.matshow(seq.T, cmap='Greens', aspect=1)
            ax.set_axis_off()
            ax.set_anchor('W')
            ax.text(-2.0,-2.0, f'{frames}', size=4)

        plt.savefig('sinusoids16.png', dpi=300, bbox_inches='tight')
        print('plotting example sinusoid batch of 16 into sinusoids16.png')
