from pathlib import Path

import torch
import torchaudio


def make_frames(wav):
    frames = torchaudio.compliance.kaldi.mfcc(wav)

    # frames = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)
    # frames += 8.
    # frames /= 4.
    return frames


class Directory(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        super().__init__()
        self.files = list(path.glob("*.wav"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.files[index])
        assert sr == 16000
        return make_frames(wav), "the quick brown fox jumps over the lazy dog"


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, url='train-clean-100'):
        super().__init__()
        self.librispeech = torchaudio.datasets.LIBRISPEECH('.', url=url, download=True)

    def __len__(self):
        return len(self.librispeech)

    def __getitem__(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        return make_frames(wav), text


def concat_datasets(s):
    if not s:
        return []
    parts = s.split(',')
    paths = [Path(part) for part in parts]
    return torch.utils.data.ConcatDataset(
        Directory(path) if path.exists() else LibriSpeech(part)
        for path, part in zip(paths, parts)
    )

