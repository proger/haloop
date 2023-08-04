from pathlib import Path

import torch
import torchaudio


class LabelFile(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        super().__init__()
        self.resample = {
            16000: torch.nn.Identity(), # speech
            22050: torchaudio.transforms.Resample(orig_freq=22050), # tts data
            32000: torchaudio.transforms.Resample(orig_freq=32000), # common voice
            44100: torchaudio.transforms.Resample(orig_freq=44100), # common voice
            48000: torchaudio.transforms.Resample(orig_freq=48000), # opus
        }
        with open(path) as f:
            self.ark = dict(line.strip().split(maxsplit=1) for line in f)
            self.filenames = list(self.ark.keys())

    def __len__(self):
        return len(self.filenames)

    def utt_id(self, index):
        return self.filenames[index]

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.filenames[index])
        resample = self.resample.get(sr)
        if not resample:
            raise ValueError(f'unsupported sample rate {sr}, add a resampler to LabelFile.resample')
        return index, resample(wav), self.ark[self.filenames[index]]


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, url='train-clean-100'):
        super().__init__()
        self.librispeech = torchaudio.datasets.LIBRISPEECH('data', url=url, download=True)

    def __len__(self):
        return len(self.librispeech)

    def utt_id(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        utt_id = f'{speaker_id}-{chapter_id}-{utterance_id:04d}'
        return utt_id

    def __getitem__(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        return index, wav, text


class Mask(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def utt_id(self, index):
        return self.dataset.utt_id(index)

    def __getitem__(self, index):
        index, frames, text = self.dataset[index]

        frames = frames[None,None,:]

        frames = torchaudio.functional.mask_along_axis_iid(
            frames,
            mask_param=frames.size(-1) // 6,
            mask_value=0,
            axis=3 # frequency
        )

        frames = torchaudio.functional.mask_along_axis_iid(
            frames,
            mask_param=7,
            mask_value=0,
            axis=2 # time
        )

        return index, frames[0, 0, :], text


class Speed(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transform = torchaudio.transforms.SpeedPerturbation(16000, [0.95, 0.98, 1.0, 1.02, 1.05])

    def __len__(self):
        return len(self.dataset)

    def utt_id(self, index):
        return self.dataset.utt_id(index)

    def __getitem__(self, index):
        index, wav, text = self.dataset[index]
        return index, self.transform(wav)[0], text


class Fbank(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def utt_id(self, index):
        return self.dataset.utt_id(index)

    def __getitem__(self, index):
        index, wav, text = self.dataset[index]
        frames = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)
        return index, frames, text


class MFCC(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def utt_id(self, index):
        return self.dataset.utt_id(index)

    def __getitem__(self, index):
        index, wav, text = self.dataset[index]
        frames = torchaudio.compliance.kaldi.mfcc(wav)

        # utterance-level CMVN
        frames -= frames.mean(dim=0)
        frames /= frames.std(dim=0)
        return index, frames, text


class WordDrop(torch.utils.data.Dataset):
    def __init__(self, dataset, p_drop_words=0.4):
        super().__init__()
        self.dataset = dataset
        self.p_drop_words = p_drop_words

    def __len__(self):
        return len(self.dataset)

    def utt_id(self, index):
        return self.dataset.utt_id(index)

    def __getitem__(self, index):
        index, frames, original_text = self.dataset[index]
        generator = torch.Generator().manual_seed(index)
        text = ' '.join(w for w in original_text.split(' ') if torch.rand(1, generator=generator) > self.p_drop_words)
        if not text:
            text = original_text
        if False:
            from kaldialign import align
            hyp, _ref = list(zip(*align(text.split(), original_text.split(), '*')))
            print(index, ' '.join(h.replace(' ', '_') for h in hyp))
        return index, frames, text


def make_dataset(s):
    match s.split(':', maxsplit=1):
        case ['labels', label_file]: # over filename
            return LabelFile(Path(label_file))
        case ['head', subset]: # any
            return torch.utils.data.Subset(make_dataset(subset), range(16))
        case ['wdrop.4', subset]: # any
            return WordDrop(make_dataset(subset), p_drop_words=0.4)
        case ['wdrop.1', subset]: # any
            return WordDrop(make_dataset(subset), p_drop_words=0.1)
        case ['mask', subset]: # over spectrograms
            return Mask(make_dataset(subset))
        case ['speed', subset]: # only applies over waveforms
            return Speed(make_dataset(subset))
        case ['mfcc', subset]: # applies over waveforms
            return MFCC(make_dataset(subset))
        case ['fbank', subset]: # applies over waveforms
            return Fbank(make_dataset(subset))
        case ['sinusoids0']: # synthetic
            from ha.sinusoids import SyntheticAlignments
            return SyntheticAlignments(examples_per_bin=100000, max=100)
        case ['sinusoids1']: # synthetic
            from ha.sinusoids import SyntheticAlignments
            return SyntheticAlignments(examples_per_bin=30000, max=500)
        case ['sinusoids2']: # synthetic
            from ha.sinusoids import SyntheticAlignments
            return SyntheticAlignments(examples_per_bin=15000, max=1000)
        case ['sinusoids3']: # synthetic
            from ha.sinusoids import SyntheticAlignments
            return SyntheticAlignments(examples_per_bin=5000, max=2000)
        case ['sinusoids4']: # synthetic
            from ha.sinusoids import SyntheticAlignments
            return SyntheticAlignments(examples_per_bin=5000, max=3000)
        case ['sinusoids-eval']: # synthetic
            from ha.sinusoids import SyntheticAlignments
            return SyntheticAlignments(examples_per_bin=10, max=3000, seed_offset=100000000)
        case [subset]:
            if Path(subset).exists():
                return LabelFile(Path(subset))
            else:
                return LibriSpeech(subset)

def concat_datasets(s):
    if not s:
        return []
    parts = s.split(',')
    paths = [Path(part) for part in parts]
    return torch.utils.data.ConcatDataset(
        make_dataset(str(part))
        for path, part in zip(paths, parts)
    )


if __name__ == '__main__':
    import sys
    for i, f, t in make_dataset(sys.argv[1]):
        print(i, f.mean(dim=0), f.std(dim=0), f.shape, t)
        break
    m,s = torch.zeros(80), torch.zeros(80)
    frames = torch.cat([f for i, f, t in make_dataset(sys.argv[1])], dim=0)
    print(f.mean(dim=0))
    print(f.std(dim=0))
