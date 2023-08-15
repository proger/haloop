import argparse
from collections import Counter, defaultdict
from itertools import chain, pairwise
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
from .monitor import register_activation_stat_hooks, print_activation_stat_hooks
from .optim import LR, configure_optimizers
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

        self.models = models
        self.encoder = models['encoder']
        self.recognizer: Decodable = models['recognizer']
        self.vocab = symbol_tape.make_vocab(args.vocab)

        self.optimizer = configure_optimizers(self, args, device_type='cuda', decay_lm_head=False)
        self.scaler = torch.cuda.amp.GradScaler()
        self.lr = LR(args)

    def load_state_dict(self, checkpoint):
        if False:
            encoder = {}
            for key in checkpoint['encoder']:
                # convert gpt-like attention to flash MHA
                if 'attn.c_attn.weight' in key:
                    l, _, _ = key.rsplit('.', maxsplit=2)
                    encoder[l + '.Wqkv.weight'] = checkpoint['encoder'][key]
                elif 'attn.c_proj.weight' in key:
                    l, _, _ = key.rsplit('.', maxsplit=2)
                    encoder[l + '.out_proj.weight'] = checkpoint['encoder'][key]
                else:
                    encoder[key] = checkpoint['encoder'][key]
        else:
            encoder = checkpoint['encoder']

        if False:
            # convert from flash MHA to my implementation
            recognizer = {}
            for key in checkpoint['recognizer']:
                if 'rotary_emb.inv_freq' in key:
                    continue
                elif 'mix_time.Wqkv.weight' in key:
                    l, _, _ = key.rsplit('.', maxsplit=2)
                    step = checkpoint['recognizer'][key].shape[0] // 3
                    recognizer[l + '.q.weight'] = checkpoint['recognizer'][key][0*step:1*step, :]
                    recognizer[l + '.k.weight'] = checkpoint['recognizer'][key][1*step:2*step, :]
                    recognizer[l + '.v.weight'] = checkpoint['recognizer'][key][2*step:3*step, :]
                elif 'mix_time.out_proj.weight' in key:
                    l, _, _ = key.rsplit('.', maxsplit=2)
                    recognizer[l + '.proj.weight'] = checkpoint['recognizer'][key]
                else:
                    recognizer[key] = checkpoint['recognizer'][key]
        else:
            recognizer = checkpoint['recognizer']

        self.encoder.load_state_dict(encoder)
        self.recognizer.load_state_dict(recognizer)
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def make_state_dict(self, **extra):
        return {
            'encoder': self.encoder.state_dict(),
            'recognizer': self.recognizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loop_args': self.args,
        } | extra

    def forward(self, inputs, targets, input_lengths, target_lengths, drop_labels=False):
        device = next(self.encoder.parameters()).device

        inputs = inputs.to(device) # (N, T, C)
        input_lengths = input_lengths.to(device) # (N,)
        targets = targets.to(device) # (N, U)
        target_lengths = target_lengths.to(device) # (N,)

        #log(inputs, targets) # works best with --batch-size 1

        measure_entropy = self.args.entropy and not self.training

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            features, feature_lengths, stats1 = self.encoder(
                inputs, input_lengths,
                measure_entropy=measure_entropy
            )
            loss, stats = self.recognizer(
                features, targets, feature_lengths, target_lengths,
                star_penalty=self.args.star_penalty,
                measure_entropy=measure_entropy,
                drop_labels=drop_labels,
            )
            if measure_entropy:
                for k in stats1:
                    print('encoder', k, torch.stack(stats1[k]))
                for k in stats:
                    print('recognizer', k, torch.stack(stats[k]))

        return loss, features, feature_lengths

    def train_one_epoch(self, epoch, global_step, train_loader, valid_loader):
        optimizer, scaler = self.optimizer, self.scaler

        optimizer.zero_grad()

        train_loss = 0.
        t0 = time.time()
        local_step, accumulate = 0, 0

        self.models.train()
        for i, (_batch_indices, inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            loss, _, _ = self.forward(inputs, targets, input_lengths, target_lengths, drop_labels=True)

            if torch.isnan(loss):
                log(f'[{epoch}, {global_step:5d}], loss is nan, skipping batch', flush=True)
                #scaler.update()
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
            grad_norm = torch.nn.utils.clip_grad_norm_(chain(self.encoder.parameters()), self.args.clip_grad_norm, error_if_nonfinite=False)
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                log(f'[{epoch}, {global_step:5d}], grad_norm is inf or nan, skipping batch, loss: {loss:.5f}, data: {_batch_indices}', flush=True)
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

            if local_step % self.args.evaluate_every == 0:
                self.evaluate(epoch, valid_loader, attempts=1)
                self.models.train()

        return global_step

    @torch.inference_mode()
    def evaluate(self, epoch, loader, attempts=1):
        valid_loss = 0.
        label_errors = Counter()
        word_errors = Counter()

        if attempts > 1:
            self.models.train()
            est_word_errors = Counter()
        else:
            self.models.eval()

        hook_handles = register_activation_stat_hooks(self.models)

        for i, (dataset_indices, inputs, targets, input_lengths, target_lengths) in enumerate(loader):
            loss, features, feature_lengths = self.forward(inputs, targets, input_lengths, target_lengths, drop_labels=False)
            if i == 0:
                print_activation_stat_hooks(self.models)
                for hook_handle in hook_handles:
                    hook_handle.remove()

            collected_hypotheses = defaultdict(list)
            gt_wer = {}

            for attempt in range(attempts):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    hypotheses, alignments = self.recognizer.decode(features, feature_lengths, target_lengths)

                valid_loss += loss.item()

                for dataset_index, ref, ref_len, hyp_, ali_, feat_len in zip(
                    dataset_indices, targets, target_lengths, hypotheses, alignments, feature_lengths
                ):
                    k = dataset_index.item()
                    label_error, word_error, hyp = self.print_example(k, ref, ref_len, hyp_, ali_, feat_len, epoch=epoch, attempt=attempt)
                    label_errors += label_error
                    word_errors += word_error
                    collected_hypotheses[k].append(hyp)
                    gt_wer[k] = word_error['total'] / word_error['length']

            if attempts > 1:
                e, est_wer = self.estimate_wer(collected_hypotheses)
                est_word_errors += e

                for k in est_wer:
                    print(epoch, k, f'est-wer: {est_wer[k]:.3f}', f'gt-wer: {gt_wer[k]:.3f}', sep="\t", flush=True)

        count = i + 1
        ler = round(label_errors['total'] / label_errors['length'], 3)
        wer = round(word_errors['total'] / word_errors['length'], 3)
        log(f'valid [{epoch}, {i + 1:5d}] loss: {valid_loss / count:.3f} ler: {ler:.3f} wer: {wer:.3f}', flush=True)
        if attempts > 1:
            est_wer = round(est_word_errors['total'] / est_word_errors['length'], 3)
            log(f'valid [{epoch}, {i + 1:5d}] estimated-wer: {est_wer:.3f} diff-wer: {wer - est_wer:.3f}', flush=True)
        if wandb.run is not None:
            wandb.log({'valid/loss': valid_loss / count, 'valid/ler': ler, 'valid/wer': wer})
        return valid_loss / count

    def estimate_wer(self, hypotheses):
        # estimate WER from multiple hypotheses
        est_word_errors = Counter()
        est_wer = {}
        for k in hypotheses:
            errors, lengths, counts = 0, 0, 0
            for l, r in pairwise(hypotheses[k]):
                errors += edit_distance(l, r)['total']
                lengths += len(r)
                counts += 1
            est_word_errors += Counter({'total': errors / counts, 'length': lengths / counts})
            est_wer[k] = errors / lengths
        return est_word_errors, est_wer

    def print_example(self, dataset_index, ref, ref_len, hyp_, ali_, feat_len, epoch, attempt=0):
        stat = {}
        hyp = hyp_.cpu().tolist()
        ali = ali_[:feat_len].cpu().tolist() if ali_ is not None else []
        ref = ref[:ref_len].cpu().tolist()

        hyp1, hyp_words = self.vocab.decode(hyp)
        ref1, ref_words = self.vocab.decode(ref)

        stat |= edit_distance(hyp1, ref1)
        stat['length'] = len(ref1)
        ler = stat['total'] / stat['length']
        stat['ler'] = round(ler, 2)
        label_error = Counter(stat)

        word_dist = edit_distance(hyp_words, ref_words)
        word_dist['length'] = len(ref_words)
        wer = word_dist['total'] / word_dist['length']
        stat['wer'] = round(wer, 2)
        word_error = Counter(word_dist)

        if self.args.quiet:
            return label_error, word_error

        ali, _ = self.vocab.decode(ali)

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

        print(epoch, dataset_index, f'hyp{attempt}', self.vocab.format(hyp), sep="\t", flush=True)
        print(epoch, dataset_index, 'ref', self.vocab.format(ref), sep="\t", flush=True)
        if ali:
            print(epoch, dataset_index, f'ali{attempt}', self.vocab.format(ali), sep="\t", flush=True)
        print(epoch, dataset_index, f'stat{attempt}', ' '.join(f'{k}={stat[k]}' for k in stat), sep="\t", flush=True)

        return label_error, word_error, hyp


def make_parser():
    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=Formatter)
    parser.add_argument('--init', type=Path, nargs='+', help="Path to checkpoint(s) to initialize from")
    parser.add_argument('--reset', action='store_true', help="Reset checkpoint epoch count (useful for LR scheduling)")
    parser.add_argument('--arch', type=str, default='transformer:512', help=create_model.__doc__)
    parser.add_argument('--vocab', type=str, default='ascii', help="Vocabulary to use: bytes|ascii|cmu|xen|path/to/words.txt")
    parser.add_argument('--compile', action='store_true', help="torch.compile the model (produces incompatible checkpoints)")
    parser.add_argument('--device', type=str, default='cuda:1', help="torch device to use")

    parser.add_argument('--exp', type=Path, default='exp/haloop', help="Path to checkpoint directory")
    parser.add_argument('--save', type=str, default='last+best', choices=['all', 'last+best', 'best'], help='What checkpoints to save after evaluation')
    parser.add_argument('--log-interval', type=int, default=100, help="Number of batches between printing training status")

    parser.add_argument('--num-epochs', type=int, default=30, help="Number of epochs to train for")
    parser.add_argument('--batch-size', type=int, default=48, help="Batch size")
    parser.add_argument('--accumulate', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--seed', type=int, default=42, help="Initial random seed")
    parser.add_argument('--entropy', action='store_true', help="Estimate decoder attention entropy at evaluation (slow)")
    parser.add_argument('--anomaly', action='store_true', help="Detect NaN/Inf during training")

    LR.add_arguments(parser)

    parser.add_argument('--star-penalty', type=float, default=None, help="Star penalty for Star CTC. If None, train with regular CTC")
    parser.add_argument('--clip-grad-norm', type=float, default=0.1, help="Clip gradient norm to this value")

    parser.add_argument('--train', type=str, help="Datasets to train on, comma separated")
    parser.add_argument('--eval', type=str, help="Datasets to evaluate on, comma separated")
    parser.add_argument('--evaluate-every', type=int, default=10000, help="Evaluate every this many steps during the training epoch")
    parser.add_argument('--test', type=str, required=False, help="Datasets to run final evaluation (test) on, comma separated")
    parser.add_argument('--test-attempts', type=int, default=1, help="Estimate WER from this many pairwise hypotheses obtained by test-time dropout (try 10?))")

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

    if args.eval:
        valid_loader = torch.utils.data.DataLoader(
            concat_datasets(args.eval),
            collate_fn=Collator(system.vocab),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            concat_datasets(args.test),
            collate_fn=Collator(system.vocab),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    epoch, global_step = 0, 0
    if args.init:
        checkpoint = torch.load(args.init[0], map_location=args.device)
        system.load_state_dict(checkpoint)
        if len(args.init) > 1:
            log('averaging models')
            avg_model = torch.optim.swa_utils.AveragedModel(system)
            for m in args.init[1:]:
                checkpoint = torch.load(m, map_location=args.device)
                system.load_state_dict(checkpoint)
                avg_model.update_parameters(system)
            system = avg_model.module

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

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

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

        checkpoint = Checkpointer(path=args.exp, save=args.save)

        for epoch in range(epoch, args.num_epochs):
            global_step = system.train_one_epoch(epoch, global_step, train_loader, valid_loader)
            valid_loss = system.evaluate(epoch, valid_loader)
            checkpoint(loss=valid_loss, epoch=epoch, checkpoint_fn=lambda: system.make_state_dict(**{
                'best_valid_loss': valid_loss,
                'epoch': epoch,
                'global_step': global_step,
            }))
    elif args.eval:
        system.evaluate(epoch, valid_loader)

    if args.test:
        system.evaluate(epoch, test_loader, attempts=args.test_attempts)


if __name__ == '__main__':
    main()
