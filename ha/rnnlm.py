from ha import argparse
import math
import pathlib
from pathlib import Path
import sys

from rich.console import Console
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .symbol_tape import Vocabulary, tokenize_bytes, tokenize_chars, load_u16, SymbolTapeNoPad, tokenize_words
from .rnn import Decoder

console = Console(log_path=False, highlight=False)
def print(*args, flush=False, **kwargs):
    console.print(*args, **kwargs)


def make_dataset(
    args,
    vocab=None,
    extend_vocab=False,
):
    batch_size, bptt_len = args.batch_size, args.bptt_len
    match str(args.train).rsplit(':', maxsplit=1):
        case ['u16', path]:
            vocab = Vocabulary(pad_token=0)
            vocab.id_to_string = {}
            vocab.string_to_id = {}
            for x in range(int(args.vocab)):
                vocab.add_new_word(str(x))
            data = load_u16(path)
            return SymbolTapeNoPad(data, batch_size=batch_size, bptt_len=bptt_len), vocab
        case ['bytes', path]:
            data, vocab = tokenize_bytes(path, vocab, extend_vocab=extend_vocab)
            dataset = SymbolTapeNoPad(data, batch_size=batch_size, bptt_len=bptt_len)
            return dataset, vocab
        case ['words', path]:
            assert isinstance(args.vocab, str), "vocab should be a file with vocabulary entries, one per line"
            data, vocab = tokenize_words(args.vocab, vocab, extend_vocab=extend_vocab)
            dataset = SymbolTapeNoPad(data, batch_size=batch_size, bptt_len=bptt_len)
            return dataset, vocab
        case ['chars', path] | [path]:
            data, vocab = tokenize_chars(path, vocab, extend_vocab=extend_vocab)
            dataset = SymbolTapeNoPad(data, batch_size=batch_size, bptt_len=bptt_len)
            return dataset, vocab


class System:
    def __init__(self, args):
        self.vocab = None

        if args.init:
            try:
                torch.serialization.safe_globals([pathlib._local.PosixPath])
            except AttributeError:
                torch.serialization.safe_globals([pathlib.PosixPath])

            checkpoint = torch.load(args.init, weights_only=False)
            self.vocab = Vocabulary()
            self.vocab.load_state_dict(checkpoint['vocab'])
            extend_vocab = False
            self.step = checkpoint.get('step', 0)
        else:
            extend_vocab = True
            self.step = 0

        if args.reset_step is not None:
            self.step = args.reset_step

        if args.train:
            self.dataset, self.vocab = make_dataset(
                args,
                self.vocab,
                extend_vocab=extend_vocab,
            )

            self.batches = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                prefetch_factor=2,
                drop_last=False
            )

        if not self.vocab:
            self.vocab = Vocabulary.bytes()

        vocab_size = len(self.vocab.id_to_string)

        self.model = Decoder(vocab_size=vocab_size,
                             emb_dim=args.rnn_size,
                             hidden_dim=args.rnn_size,
                             num_layers=args.num_layers,
                             dropout=args.dropout)
        self.model = self.model.to(args.device)

        if args.init:
            self.model.load_state_dict(checkpoint['model'])

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
        if args.init:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if args.init:
            self.state = checkpoint['state'][:, :args.batch_size]  # truncate to the test batch size
            self.prompt = checkpoint['prompt'][:, :args.batch_size]
        else:
            self.state = self.model.init_hidden(args.batch_size)
            self.prompt = torch.zeros((1, args.batch_size), dtype=torch.long, device=args.device)

        self.log_interval = args.log_interval
        self.args = args

    def make_state_dict(self):
        return {
            'args': vars(self.args),
            'vocab': self.vocab.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'state': self.state,
            'prompt': self.prompt,
        }

    def prepare_prompt(self, prompt):
        device = next(self.model.parameters()).device

        prompt_list = [self.vocab.string_to_id[char] if isinstance(char, str) else char for char in prompt]
        x = torch.tensor(prompt_list).to(device).unsqueeze(1).long()

        return x, self.model.init_hidden()

    @torch.inference_mode()
    def complete(self, prompt, steps=512, top_k=1):
        model = self.model
        model.eval()

        x, states = self.prepare_prompt(prompt)

        # generate first token distribution, compute probability of the prompt
        logits, states = model(x, states)
        prompt_logits = nn.functional.cross_entropy(logits[:-1], x[1:].view(-1), reduction='none').sum()
        prompt_logits_base2 = prompt_logits / math.log(2)
        prompt_bits_per_token = prompt_logits_base2 / len(x[1:])
        out_list = self.sample(logits, states, steps=steps, topk=topk)
        return prompt_bits_per_token, joiner.join(out_list)

    def sample(self, logits, states, steps=512, top_k=1):
        if steps <= 0:
            return []

        model = self.model

        out_list = []
        joiner = ''
        def cast(s):
            nonlocal joiner
            if isinstance(s, int):
                joiner = b''
                return s.to_bytes(1, 'big')
            elif isinstance(s, bytes):
                joiner = b''
                return s
            return s

        # sample first token
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        probs = probs[-1]

        ix = probs.multinomial(num_samples=1)

        # output first token only if we're asked to generate any samples
        out_list.append(cast(self.vocab.id_to_string[int(ix)]))
        x = ix.unsqueeze(1)

        # generate remaining tokens
        for k in range(steps-1):
            logits, states = model(x, states)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            ix = probs.multinomial(num_samples=1)
            out_list.append(cast(self.vocab.id_to_string[int(ix)]))
            x = ix

        return joiner.join(out_list)

    def train_one_epoch(self):
        model, batches = self.model, self.batches
        optimizer = self.optimizer

        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = self.state
        prompt = self.prompt.to(self.args.device)
        hyp = ''
        matches, insertions, total = 0, 0, 0

        for i, batch in enumerate(batches):
            if self.step > i:
                continue

            batch = batch.to(self.args.device).long().squeeze(0)
            input = torch.cat([prompt, batch[:-1]], dim=0)
            prompt = batch[-1:]

            output, state = model(input, state)
            loss = F.cross_entropy(output, batch.reshape(-1), ignore_index=0, reduction='mean')
            state = model.truncate_hidden(state)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if i % self.log_interval == 0:
                train_bpc = loss.item() / math.log(2)

                def erase(count):
                    for _ in range(count):
                        sys.stdout.write("\b \b")  # backspace + space + backspace (to erase)

                if self.args.hyp or self.args.chunk:
                    ref = self.vocab.decode(input.squeeze(0).tolist())[0]

                    def longest_common_prefix(a, b):
                        i = 0
                        while i < min(len(a), len(b)) and a[i] == b[i]:
                            i += 1
                        return a[:i], a[i:], b[i:]

                    matched, delete, insert = longest_common_prefix(hyp, ref)

                    matches += len(matched)
                    insertions += len(insert)
                    total += len(ref)

                    if isinstance(ref, bytes):
                        matched = ''.join([f'{x:02x}' for x in list(matched)])
                        delete = ''.join([f'{x:02x}' for x in list(delete)])
                        insert = ''.join([f'{x:02x}' for x in list(insert)])

                    if self.args.chunk:
                        # do not print deletions in chunking mode, we don't care of the network prediction
                        print(f"[cyan]{matched}[/cyan][magenta]{insert}[/magenta]", end='')
                    else:  # self.args.hyp
                        print(f"[cyan]{matched}[/cyan][magenta]{delete}[/magenta]{insert}", end='')

                    with torch.inference_mode():
                        model.eval()
                        hyp = self.sample(output, state, steps=self.args.bptt_len, top_k=self.args.top_k)
                    model.train()
                    #print(f"{input_text}[blue]{hyp}[/blue] [dim]ppl: {loss.exp().item():.3f} step {i}/{len(batches)}", end='')
                else:
                    _, eval_outputs = self.evaluate()

                    print(f"step {i}/{len(batches)} loss: {loss.item():.3f} ppl: {loss.exp().item():.3f} bpc: {train_bpc:.3f} grad_norm: {grad_norm.item():.3f} {'; '.join(eval_outputs)}")

                model.train()
            wandb.log({'train/loss': loss.item(),
                       'train/ppl': loss.exp().item(),
                       'train/lr': self.args.lr,
                       'train/grad_norm': grad_norm.item()})

            self.state = state
            self.prompt = prompt

            if self.step % self.args.save_interval == 0:
                self.save()

            self.step = i + 1

            if self.args.max_steps >= 0 and i == self.args.max_steps:
                break

        if self.args.chunk:
            with open(self.args.chunk, 'w') as f:
                f.write(f"matches {matches}\n")
                f.write(f"insertions {insertions}\n")
                f.write(f"total {total}\n")

        return self.step

    def evaluate(self):
        prompt_scores = []
        outputs = []

        def prompt_stream():
            for prompt in (self.args.complete or []):
                yield self.args.start_token + prompt
            for prompt_file in (self.args.complete_file or []):
                with open(prompt_file) as f:
                    for line in f:
                        utterance_id, text = line.strip().split(maxsplit=1)
                        yield self.args.start_token + text

        for prompt in prompt_stream():
            if self.args.vocab != 'auto':
                prompt = prompt.encode('utf-8')
            prompt_score, completion = self.complete(prompt, self.args.bptt_len, top_k=self.args.top_k)
            output = prompt + completion if completion else prompt
            if self.args.vocab != 'auto':
                outputs.append(str(output, 'utf-8', errors='replace'))
            else:
                outputs.append(output)
            prompt_scores.append(prompt_score.item())

        return torch.tensor(prompt_scores), outputs

    def save(self):
        save = None
        if self.args.save:
            save = self.args.save.format(epoch=self.epoch, step=self.step)

        if save:
            print('saving', save)
            torch.save(self.make_state_dict(), save)


def main():
    parser = argparse.ArgumentParser(description="hal trains recurrent language models",
                                     formatter_class=argparse.Formatter, epilog="""\
To train a RNN on characters:
% hal --train bruk.txt --hyp

To train a RNN on bytes:
% hal --train bytes:bruk.txt --hyp

To train a RNN on 16-bit words (like https://huggingface.co/datasets/darkproger/uk4b/tree/main):
% hal --train u16:bruk.txt --hyp

To produce 10-token completions of two strings try:
% hal --init librispeech-1024.pt --rnn-size 1024 --bptt-len 10 --complete "IS THIS A BIRD" "IS THIS A PLANE"

To compute BPC on evaluation data from files (first column is ignored) try:
% hal --init librispeech-1024.pt --bptt-len 0 --rnn-size 1024 --complete-file LibriSpeech/dev-clean/*/*/*.txt

␄
""")
    parser.add_argument('--init', type=Path, help="Path to checkpoint to initialize from")
    parser.add_argument('--reset-step', type=int, help="Rewind data to this step")
    parser.add_argument('--save', type=str, default='rnnlm-epoch{epoch}-step{step}.pt', help="Path to save checkpoint to. Substitutes {epoch} and {step} using str.format")
    parser.add_argument('--save-interval', type=int, default=100000, help="Save interval in steps in addition to once every epoch")
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--lr', default=0.002, type=float, help='AdamW learning rate')
    parser.add_argument('--wd', default=0.1, type=float, help='AdamW weight decay')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    parser.add_argument('--epochs', default=1, type=int, help='number of passes over the training data')
    parser.add_argument('--max-steps', default=-1, type=int, help='maximum number of training steps (useful for e.g. lr search)')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--bptt-len', default=64, type=int, help='RNN sequence length (window size)')
    parser.add_argument('--rnn-size', default=512, type=int, help='RNN width')
    parser.add_argument('--num-layers', default=1, type=int, help='RNN depth')
    parser.add_argument('--vocab', default='auto', type=str, help='how to build vocabulary')
    parser.add_argument('--train', type=str, help='Train model on this data')
    parser.add_argument('--top-k', type=int, default=1, help='top-k sampling')
    parser.add_argument('--log-interval', type=int, default=1, help="Number of batches between printing training status")
    parser.add_argument('--hyp', action='store_true', help="Continue the training data for bptt_len steps for visualization. Supersedes --complete/--complete-file.")
    parser.add_argument('--chunk', type=str, help="Chunk the output using the principle of history compression and store the counts to file. Supersedes --hyp.")
    parser.add_argument('--complete', type=str, nargs='+', help="Prompts to complete during evaluation")
    parser.add_argument('--start-token', type=str, default='\n', help="Prepend this token to every prompt. This token is necessary to compute p(prompt|start-token)")
    parser.add_argument('--complete-file', type=Path, nargs='+', help="Prompts to complete during evaluation as a file. First column is utterance id.")
    parser.add_argument('--num-workers', type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args()

    torch.manual_seed(3407)

    self = System(args)

    if args.train:
        print(args)
        wandb.init(config=args)

        for epoch in range(args.epochs):
            self.epoch = epoch

            try:
                self.train_one_epoch()
            except KeyboardInterrupt:
                pass
            finally:
                self.save()

            self.step = 0  # reset step counter for the new epoch

    prompt_scores, outputs = self.evaluate()
    if prompt_scores.numel():
        for prompt_score, output in zip(prompt_scores, outputs):
            print('{:.2f}'.format(prompt_score), 'bpc', output)
        print('mean bpc', torch.mean(prompt_scores).item())

if __name__ == '__main__':
    main()
