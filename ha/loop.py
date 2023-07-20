import argparse
from collections import Counter
from itertools import chain
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data
from kaldialign import edit_distance, align
import wandb

from .data import concat_datasets
from .init import create_model
from .recognizer import Decodable
from . import symbol_tape
from .lr import LR
from .checkpoint import Checkpointer


def log(*args, flush=False, **kwargs):
    print(*args, **kwargs, flush=flush, file=sys.stderr)


class Collator:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        batch_indices = torch.tensor([b[0] for b in batch])
        input_lengths = torch.tensor([len(b[1]) for b in batch])
        inputs = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True)
        targets = [self.vocab.encode(b[2]) for b in batch]
        target_lengths = torch.tensor([len(t) for t in targets])
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        return batch_indices, inputs, targets, input_lengths, target_lengths


class System(nn.Module):
    def __init__(self, args, models):
        super().__init__()
        self.args = args

        self.encoder = models['encoder']
        self.recognizer: Decodable = models['recognizer']
        self.vocab = symbol_tape.make_vocab(args.vocab)

        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.recognizer.parameters()))
        self.scaler = torch.cuda.amp.GradScaler()
        self.lr = LR(args)

    def load_state_dict(self, checkpoint):
        encoder = {}
        for key in checkpoint['encoder']:
            if 'attn.c_attn.weight' in key:
                l, _, _ = key.rsplit('.', maxsplit=2)
                encoder[l + '.Wqkv.weight'] = checkpoint['encoder'][key]
            elif 'attn.c_proj.weight' in key:
                l, _, _ = key.rsplit('.', maxsplit=2)
                encoder[l + '.out_proj.weight'] = checkpoint['encoder'][key]
            else:
                encoder[key] = checkpoint['encoder'][key]

        self.encoder.load_state_dict(encoder, strict=False)
        self.recognizer.load_state_dict(checkpoint['recognizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'vocab' in checkpoint:
            log('loading vocab state')
            self.vocab.load_state_dict(checkpoint['vocab'])

    def make_state_dict(self, **extra):
        return {
            'encoder': self.encoder.state_dict(),
            'recognizer': self.recognizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'vocab': self.vocab.state_dict(),
            'loop_args': self.args,
        } | extra

    def subsampled_lengths(self, input_lengths):
        if hasattr(self.encoder, 'subsampled_lengths'):
            return self.encoder.subsampled_lengths(input_lengths)
        else:
            return input_lengths

    def forward(self, inputs, targets, input_lengths, target_lengths):
        device = next(self.encoder.parameters()).device

        inputs = inputs.to(device) # (N, T, C)
        input_lengths = input_lengths.to(device) # (N,)
        targets = targets.to(device) # (N, U)
        target_lengths = target_lengths.to(device) # (N,)

        #log(inputs, targets) # works best with --batch-size 1

        feature_lengths = self.subsampled_lengths(input_lengths)
        measure_entropy = self.args.entropy and not self.training

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            features = self.encoder(inputs, input_lengths)
            loss, stats = self.recognizer(
                features, targets, feature_lengths, target_lengths,
                star_penalty=self.args.star_penalty,
                measure_entropy=measure_entropy,
            )
            if measure_entropy:
                for k in stats:
                    print(k, torch.stack(stats[k]))

        return loss, features, feature_lengths

    def train_one_epoch(self, epoch, global_step, train_loader):
        optimizer, scaler = self.optimizer, self.scaler

        optimizer.zero_grad()

        train_loss = 0.
        t0 = time.time()
        local_step, accumulate = 0, 0

        self.train()
        self.encoder.train()
        self.recognizer.train()
        for i, (_batch_indices, inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            loss, _, _ = self.forward(inputs, targets, input_lengths, target_lengths)

            if torch.isnan(loss):
                log(f'[{epoch}, {global_step:5d}], loss is nan, skipping batch', flush=True)
                scaler.update()
                continue

            if torch.isinf(loss):
                log(f'[{epoch}, {global_step:5d}], loss is inf, skipping batch, skipping scaler update', flush=True)
                continue

            loss = loss / self.args.accumulate
            scaler.scale(loss).backward()
            accumulate += 1

            if accumulate % self.args.accumulate:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(chain(self.encoder.parameters(), self.recognizer.parameters()), self.args.clip_grad_norm)
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                log(f'[{epoch}, {global_step:5d}], grad_norm is inf or nan, skipping batch', flush=True)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            lr = self.lr.apply_lr_(optimizer, global_step)
            global_step, local_step = global_step + 1, local_step + 1

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() / self.args.log_interval
            if local_step % self.args.log_interval:
                continue

            t1 = time.time()
            log(f'[{epoch}, {global_step:5d}] time: {t1-t0:.3f} loss: {train_loss:.3f} grad_norm: {grad_norm:.3f} lr: {lr:.5f}', flush=True)
            wandb.log({'train/loss': train_loss, 'train/grad_norm': grad_norm, 'train/lr': lr, 'iter': global_step})
            t0 = t1
            train_loss = 0.

        return global_step

    @torch.inference_mode()
    def evaluate(self, epoch, valid_loader):
        valid_loss = 0.
        label_errors = Counter()
        word_errors = Counter()

        self.eval()
        self.encoder.eval()
        self.recognizer.eval()
        for i, (dataset_indices, inputs, targets, input_lengths, target_lengths) in enumerate(valid_loader):
            loss, features, feature_lengths = self.forward(inputs, targets, input_lengths, target_lengths)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                hypotheses, alignments = self.recognizer.decode(features, feature_lengths, target_lengths)

            valid_loss += loss.item()

            for dataset_index, ref, ref_len, hyp_, ali_, feat_len in zip(
                dataset_indices, targets, target_lengths, hypotheses, alignments, feature_lengths
            ):
                stat = {}
                hyp = hyp_.cpu().tolist()
                ali = ali_[:feat_len].cpu().tolist() if ali_ is not None else []
                ref = ref[:ref_len].cpu().tolist()

                hyp1 = self.vocab.decode(hyp)
                ref1 = self.vocab.decode(ref)

                stat |= edit_distance(hyp1, ref1)
                stat['length'] = len(ref1)
                ler = stat['total'] / stat['length']
                stat['ler'] = round(ler, 2)
                label_errors += Counter(stat)

                ref_words = ref1.split()
                word_dist = edit_distance(hyp1.split(), ref_words)
                word_dist['length'] = len(ref_words)
                wer = word_dist['total'] / word_dist['length']
                stat['wer'] = round(wer, 2)
                word_errors += Counter(word_dist)

                if self.args.quiet:
                    continue

                ali = self.vocab.decode(ali)

                if isinstance(ref1, list):
                    star = '␣'
                    hyp, ref = list(zip(*align(hyp1, ref1, star)))
                    ali = tuple(ali)
                elif isinstance(ref1, str):
                    star = '␣'
                    hyp, ref = list(zip(*align(hyp1, ref1, star)))
                    hyp, ref = ''.join(hyp), ''.join(ref)
                else:
                    star = 42 # b'*'
                    hyp, ref = list(zip(*align(hyp1, ref1, star)))
                    hyp, ref = bytes(hyp), bytes(ref)

                dataset_index = dataset_index.item()
                print(epoch, dataset_index, 'hyp', self.vocab.format(hyp), sep="\t", flush=True)
                print(epoch, dataset_index, 'ref', self.vocab.format(ref), sep="\t", flush=True)
                if ali:
                    print(epoch, dataset_index, 'ali', self.vocab.format(ali), sep="\t", flush=True)
                print(epoch, dataset_index, 'stat', ' '.join(f'{k}={stat[k]}' for k in stat), sep="\t", flush=True)

        count = i + 1
        ler = round(label_errors['total'] / label_errors['length'], 3)
        wer = round(word_errors['total'] / word_errors['length'], 3)
        log(f'valid [{epoch}, {i + 1:5d}] loss: {valid_loss / count:.3f} ler: {ler:.3f} wer: {wer:.3f}', flush=True)
        if wandb.run is not None:
            wandb.log({'valid/loss': valid_loss / count, 'valid/ler': ler, 'valid/wer': wer})
        return valid_loss / count


def make_parser():
    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=Formatter)
    parser.add_argument('--init', type=Path, help="Path to checkpoint to initialize from")
    parser.add_argument('--reset', action='store_true', help="Reset checkpoint epoch count (useful for LR scheduling)")
    parser.add_argument('--arch', type=str, default='recognizer:lstm:128', help=create_model.__doc__)
    parser.add_argument('--vocab', type=str, default='ascii', help="Vocabulary to use: bytes|ascii|cmu|xen|words:path/to/words.txt")
    parser.add_argument('--compile', action='store_true', help="torch.compile the model (produces incompatible checkpoints)")
    parser.add_argument('--device', type=str, default='cuda:1', help="torch device to use")

    parser.add_argument('--exp', type=Path, default='exp/haloop', help="Path to checkpoint directory")
    parser.add_argument('--save-all', action='store_true', help='Always save a checkpoint after each evaluation')
    parser.add_argument('--log-interval', type=int, default=100, help="Number of batches between printing training status")

    parser.add_argument('--num-epochs', type=int, default=30, help="Number of epochs to train for")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--accumulate', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--seed', type=int, default=3407, help="Initial random seed")
    parser.add_argument('--entropy', action='store_true', help="Estimate decoder attention entropy at evaluation (slow)")

    LR.add_arguments(parser)

    parser.add_argument('--star-penalty', type=float, default=None, help="Star penalty for Star CTC. If None, train with regular CTC")
    parser.add_argument('--clip-grad-norm', type=float, default=0.1, help="Clip gradient norm to this value")

    parser.add_argument('--train', type=str, help="Datasets to train on, comma separated")
    parser.add_argument('--eval', type=str, default='dev-clean', help="Datasets to evaluate on, comma separated")
    parser.add_argument('--test', type=str, default='dev-clean', help="Datasets to run final evaluation on, comma separated")
    parser.add_argument('-q', '--quiet', action='store_true', help="Only print evaluation summary")
    parser.add_argument('--wandb', action='store_true', help="Unconditionally log to wandb")
    parser.add_argument('--num-workers', type=int, default=32, help="Number of workers for data loading")
    return parser


def main():
    args = make_parser().parse_args()
    log(args)

    torch.manual_seed(args.seed)

    models = create_model(args.arch, compile=False).to(args.device)
    system = System(args, models)

    valid_loader = torch.utils.data.DataLoader(
        concat_datasets(args.eval),
        collate_fn=Collator(system.vocab),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    epoch, global_step = 0, 0
    if args.init:
        checkpoint = torch.load(args.init, map_location=args.device)
        system.load_state_dict(checkpoint)
        if not args.reset:
            epoch = checkpoint.get('epoch', -1) + 1
            global_step = checkpoint.get('global_step', -1) + 1
    else:
        log('initializing randomly')

    if args.compile:
        system = torch.compile(system, mode='reduce-overhead')

    log('model parameters', sum(p.numel() for p in system.parameters() if p.requires_grad))

    if args.train or args.wandb:
        wandb.init(project='ha', config=args, name=str(args.exp))

    if args.train:
        train_loader = torch.utils.data.DataLoader(
            concat_datasets(args.train),
            collate_fn=Collator(system.vocab),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )

        log('total training minibatches:', len(train_loader) * args.num_epochs)

        checkpoint = Checkpointer(path=args.exp, save_all=args.save_all)

        for epoch in range(epoch, args.num_epochs):
            global_step = system.train_one_epoch(epoch, global_step, train_loader)
            valid_loss = system.evaluate(epoch, valid_loader)
            checkpoint(loss=valid_loss, epoch=epoch, checkpoint_fn=lambda: system.make_state_dict(**{
                'best_valid_loss': valid_loss,
                'epoch': epoch,
                'global_step': global_step,
            }))
    else:
        system.evaluate(epoch, valid_loader)

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            concat_datasets(args.test),
            collate_fn=Collator(system.vocab),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        system.evaluate(epoch, test_loader)


if __name__ == '__main__':
    main()
