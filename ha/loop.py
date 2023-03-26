import argparse
import time
from itertools import chain
from pathlib import Path

from rich.console import Console
import torch
import torch.nn as nn
import torch.utils.data
from kaldialign import edit_distance, align
import wandb

from .data import concat_datasets
from .beam import ctc_beam_search_decode_logits
from .model import Encoder, CTCRecognizer, StarRecognizer
from .xen import Vocabulary


console = Console()
def print(*args, flush=False, **kwargs):
    console.log(*args, **kwargs)

class Collator:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, batch):
        input_lengths = torch.tensor([len(b[0]) for b in batch])
        inputs = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        targets = [self.vocabulary.encode(b[1]) for b in batch]
        target_lengths = torch.tensor([len(t) for t in targets])
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        return inputs, targets, input_lengths, target_lengths


class System(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder().to(args.device)
        self.vocabulary = Vocabulary(args.glottal_closures)
        if args.star_penalty is not None:
            self.recognizer = StarRecognizer(star_penalty=args.star_penalty,
                                             vocab_size=len(self.vocabulary)).to(args.device)
        else:
            self.recognizer = CTCRecognizer(vocab_size=len(self.vocabulary)).to(args.device)
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.recognizer.parameters()), lr=args.lr)
        self.scaler = torch.cuda.amp.GradScaler()

    def load_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.recognizer.load_state_dict(checkpoint['recognizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def state_dict(self, **extra):
        return {
            'encoder': self.encoder.state_dict(),
            'recognizer': self.recognizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        } | extra

    def train_one_epoch(self, epoch, train_loader):
        args, device = self.args, self.args.device
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

            if torch.isinf(loss):
                print(f'[{epoch + 1}, {i + 1:5d}], loss is inf, skipping batch, skipping scaler update', flush=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(chain(encoder.parameters(), recognizer.parameters()), 0.1)
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                print(f'[{epoch + 1}, {i + 1:5d}], grad_norm is inf or nan, skipping batch', flush=True)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            if i and i % args.log_interval == 0:
                train_loss = train_loss / args.log_interval
                t1 = time.time()
                print(f'[{epoch + 1}, {i + 1:5d}] time: {t1-t0:.3f} loss: {train_loss:.3f} grad_norm: {grad_norm:.3f}', flush=True)
                wandb.log({'train/loss': train_loss, 'train/grad_norm': grad_norm})
                t0 = t1

    @torch.inference_mode()
    def evaluate(self, epoch, valid_loader):
        device = self.args.device
        encoder, recognizer, vocabulary = self.encoder, self.recognizer, self.vocabulary

        valid_loss = 0.
        lers = []

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

            if i < 10:
                outputs = recognizer.log_probs(outputs)
                for ref, ref_len, seq, hyp_len in zip(targets, target_lengths, outputs, input_lengths):
                    seq = seq[:hyp_len].cpu()
                    #print('greedy', seq.argmax(dim=-1).tolist())
                    decoded = ctc_beam_search_decode_logits(seq)
                    hyp1 = vocabulary.decode(filter(None, decoded[0][0]))
                    ref1 = vocabulary.decode(ref[:ref_len])
                    hyp, ref = list(zip(*align(hyp1, ref1, '*')))

                    dist = edit_distance(hyp1, ref1)
                    dist['length'] = len(ref1)
                    dist['ler'] = round(dist['total'] / dist['length'], 2)

                    if i == 0:
                        console.print('hyp', ' '.join(h.replace(' ', '_') for h in hyp), overflow='crop')
                        console.print('ref', ' '.join(r.replace(' ', '_') for r in ref), overflow='crop')
                        print(dist)

                    lers.append(dist['ler'])

        count = i + 1
        ler = round(sum(lers) / len(lers), 3)
        print(f'valid [{epoch + 1}, {i + 1:5d}] loss: {valid_loss / count:.3f} sample ler: {ler:.3f}', flush=True)
        if wandb.run is not None:
            wandb.log({'valid/loss': valid_loss / count, 'valid/ler': ler})
        return valid_loss / count


def main():
    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=Formatter)
    parser.add_argument('--init', type=Path, help="Path to checkpoint to initialize from")
    parser.add_argument('--save', type=Path, default='ckpt.pt', help="Path to save checkpoint to")
    parser.add_argument('--log-interval', type=int, default=100, help="Number of batches between printing training status")
    parser.add_argument('--num-epochs', type=int, default=30, help="Number of epochs to train for")
    parser.add_argument('--device', type=str, default='cuda:1', help="torch device to use")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument('--train', type=str, help="Datasets to train on, comma separated")
    parser.add_argument('--eval', type=str, default='dev-clean', help="Datasets to evaluate on, comma separated")
    parser.add_argument('--encoder', type=str, default='uni', choices=['uni', 'bi'], help="Encoder to use: unidirectional LSTM or bidirectional Transformer")
    parser.add_argument('--compile', action='store_true', help="torch.compile the model (produces incompatible checkpoints)")
    parser.add_argument('--star-penalty', type=float, default=None, help="Star penalty for Star CTC. If None, train with regular CTC")
    parser.add_argument('--num-workers', type=int, default=32, help="Number of workers for data loading")
    parser.add_argument('--glottal-closures', action='store_true', help="Add glotal closures to the vocabulary")
    args = parser.parse_args()

    print(args)

    torch.manual_seed(3407)

    system = System(args)

    valid_loader = torch.utils.data.DataLoader(
        concat_datasets(args.eval),
        collate_fn=Collator(system.vocabulary),
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.init:
        checkpoint = torch.load(args.init, map_location=args.device)
        system.load_state_dict(checkpoint)

    if args.compile:
        system = torch.compile(system, options={'trace.graph_diagram': True})

    if args.train:
        wandb.init(project='ha', config=args)

        train_loader = torch.utils.data.DataLoader(
            concat_datasets(args.train),
            collate_fn=Collator(system.vocabulary),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )

        best_valid_loss = float('inf')
        for epoch in range(args.num_epochs):
            system.train_one_epoch(epoch, train_loader)

            valid_loss = system.evaluate(epoch, valid_loader)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print('saving model', args.save)
                torch.save(system.state_dict(best_valid_loss=best_valid_loss, epoch=epoch), args.save)
    else:
        system.evaluate(-100, valid_loader)

if __name__ == '__main__':
    main()
