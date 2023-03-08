from itertools import chain
import math
import time

import torch.utils.data
import torchaudio
import torch
import torch.nn as nn


from reco import Encoder, Recognizer


log_interval = 100
num_epochs = 30
device = 'cuda:1'
recognizer = Recognizer().to(device)

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, url='train-clean-360'):
        super().__init__()
        self.librispeech = torchaudio.datasets.LIBRISPEECH('.', url=url, download=True)

    def __len__(self):
        return len(self.librispeech)
    
    def __getitem__(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        frames = torchaudio.compliance.kaldi.mfcc(wav)

        return frames, text

def collate(batch):
    input_lengths = torch.tensor([len(b[0]) for b in batch])
    inputs = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence([recognizer.encode(b[1]) for b in batch], batch_first=True, padding_value=-1) 
    target_lengths = torch.tensor([len(t) for t in targets])
    
    return inputs, targets, input_lengths, target_lengths
        
        
train_set = LibriSpeech()
train_loader = torch.utils.data.DataLoader(
    train_set,
    collate_fn=collate,
    batch_size=32,
    shuffle=True,
    num_workers=32
)
valid_set = LibriSpeech('dev-clean')
valid_loader = torch.utils.data.DataLoader(
    valid_set,
    collate_fn=collate,
    batch_size=16,
    shuffle=False,
    num_workers=32
)


encoder = Encoder().to(device)
scaler = torch.cuda.amp.GradScaler()

optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    train_loss = 0.
    t0 = time.time()
    for i, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
        inputs = inputs.to(next(encoder.parameters()).device)
        targets = targets.to(next(encoder.parameters()).device)
        input_lengths = input_lengths.to(next(encoder.parameters()).device)
        target_lengths = target_lengths.to(next(encoder.parameters()).device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = encoder(inputs)
        loss = recognizer(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        if i and i % log_interval == 0:
            train_loss = train_loss / log_interval
            t1 = time.time()
            ppl = math.exp(train_loss)
            print(f'[{epoch + 1}, {i + 1:5d}] time: {t1-t0:.3f} loss: {train_loss:.3f} ppl: {ppl:.3f} grad_norm: {grad_norm:.3f}')
            t0 = t1

    valid_loss = 0.
    with torch.inference_mode():
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(valid_loader):
            inputs = inputs.to(next(encoder.parameters()).device)
            targets = targets.to(next(encoder.parameters()).device)
            input_lengths = input_lengths.to(next(encoder.parameters()).device)
            target_lengths = target_lengths.to(next(encoder.parameters()).device)

            logits = encoder(inputs)
            loss = recognizer(logits, targets)

            valid_loss += loss.item()
    print(f'valid [{epoch + 1}, {i + 1:5d}] loss: {valid_loss / i:.3f}')


torch.save(encoder.state_dict(), 'encoder.pt')
torch.save(recognizer.state_dict(), 'recognizer.pt')
torch.save(optimizer.state_dict(), 'optimizer.pt')

