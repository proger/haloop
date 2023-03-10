import argparse
import math
import sys
import time
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
import torchaudio
from kaldialign import edit_distance

from .beam import ctc_beam_search_decode_logits
from .model import Encoder, Recognizer, Vocabulary

parser = argparse.ArgumentParser()
parser.add_argument('--init', type=Path)
parser.add_argument('--save', type=Path, default='ckpt.pt')
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--num-epochs', type=int, default=30)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--eval', action='store_true', help="Evaluate and exit")
args = parser.parse_args()


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, url='train-clean-360'):
        super().__init__()
        self.librispeech = torchaudio.datasets.LIBRISPEECH('.', url=url, download=True)

    def __len__(self):
        return len(self.librispeech)

    def __getitem__(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        frames = torchaudio.compliance.kaldi.mfcc(wav)

        # frames = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)
        # frames += 8.
        # frames /= 4.

        return frames, text

vocabulary = Vocabulary()

def collate(batch):
    input_lengths = torch.tensor([len(b[0]) for b in batch])
    inputs = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
    targets = [vocabulary.encode(b[1]) for b in batch]
    target_lengths = torch.tensor([len(t) for t in targets])
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
    return inputs, targets, input_lengths, target_lengths


class System:
    def __init__(self, args):
        self.args = args
        self.encoder = Encoder().to(args.device)
        self.recognizer = Recognizer().to(args.device)
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.recognizer.parameters()), lr=3e-4)
        self.scaler = torch.cuda.amp.GradScaler()
        self.vocabulary = Vocabulary()

    def load_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.recognizer.load_state_dict(checkpoint['recognizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'recognizer': self.recognizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def train(self, epoch, train_loader):
        device = self.args.device
        encoder, recognizer, optimizer, scaler = self.encoder, self.recognizer, self.optimizer, self.scaler

        optimizer.zero_grad()
        encoder.train()
        recognizer.train()

        train_loss = 0.
        t0 = time.time()
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            input_lengths = encoder.subsampled_lengths(input_lengths)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = encoder(inputs)
                loss = recognizer(outputs, targets, input_lengths, target_lengths)

            if torch.isnan(loss):
                print(f'[{epoch + 1}, {i + 1:5d}], loss is nan, skipping batch', flush=True)
                scaler.update()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(chain(encoder.parameters(), recognizer.parameters()), 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            if i and i % args.log_interval == 0:
                train_loss = train_loss / args.log_interval
                t1 = time.time()
                print(f'[{epoch + 1}, {i + 1:5d}] time: {t1-t0:.3f} loss: {train_loss:.3f} grad_norm: {grad_norm:.3f}', flush=True)
                t0 = t1

    @torch.inference_mode()
    def evaluate(self, epoch, valid_loader):
        device = self.args.device
        encoder, recognizer, vocabulary = self.encoder, self.recognizer, self.vocabulary

        valid_loss = 0.

        encoder.eval()
        recognizer.eval()
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(valid_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            input_lengths = encoder.subsampled_lengths(input_lengths)

            outputs = encoder(inputs)
            loss = recognizer(outputs, targets, input_lengths, target_lengths)

            valid_loss += loss.item()

            if i == 0:
                for ref, ref_len, seq, hyp_len in zip(targets, target_lengths, outputs, input_lengths):
                    seq = recognizer.log_probs(seq)[:hyp_len]
                    #print('greedy', seq.argmax(dim=-1).tolist())
                    decoded = ctc_beam_search_decode_logits(seq)
                    #print(decoded)
                    hyp = vocabulary.decode(decoded[0][0])
                    ref = vocabulary.decode(ref[:ref_len])
                    print('hyp', hyp)
                    print('ref', ref)
                    print('err', edit_distance(hyp, ref))
                    print()

        print(f'valid [{epoch + 1}, {i + 1:5d}] loss: {valid_loss / i:.3f}', flush=True)
        return valid_loss / i


def main():
    train_set = LibriSpeech()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=collate,
        batch_size=16,
        shuffle=True,
        num_workers=32,
        drop_last=True
    )
    valid_set = LibriSpeech('dev-clean')
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        collate_fn=collate,
        batch_size=16,
        shuffle=False,
        num_workers=32
    )

    system = System(args)
    if args.init:
        checkpoint = torch.load(args.init, map_location=args.device)
        system.load_state_dict(checkpoint)

    if args.eval:
        system.evaluate(0, valid_loader)
        sys.exit(0)

    best_valid_loss = float('inf')
    for epoch in range(args.num_epochs):
        system.train(epoch, train_loader)

        valid_loss = system.evaluate(epoch, valid_loader)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('saving model', args.save)
            torch.save(system.state_dict(), args.save)


if __name__ == '__main__':
    main()