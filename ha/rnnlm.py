from pathlib import Path
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .symbol_tape import Vocabulary, tokenize_bytes, tokenize_chars, SymbolTape


class LM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()

        self.num_classes = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.out_layer = nn.Linear(hidden_dim, vocab_size)

        self.out_layer.weight = self.embedding.weight

    def forward(self, input, state):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        output = self.out_layer(output)
        output = output.view(-1, self.num_classes)
        return output, state

    def init_hidden(self, batch_size):
        weight = self.out_layer.weight
        h = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        c = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h,c)

    def truncate_hidden(sellf, state):
        h,c = state
        return (h.detach(), c.detach())



class System:
    def __init__(self, args):
        self.vocab = None

        if args.init:
            checkpoint = torch.load(args.init)
            self.vocab = Vocabulary()
            self.vocab.load_state_dict(checkpoint['vocab'])
            extend_vocab = False
        else:
            extend_vocab = True

        if args.train:
            if args.bytes_as_tokens:
                self.data, self.vocab = tokenize_bytes(args.train, self.vocab, extend_vocab=extend_vocab)
            else:
                self.data, self.vocab = tokenize_chars(args.train, self.vocab, extend_vocab=extend_vocab)
            self.batches = SymbolTape(self.data, args.batch_size, args.bptt_len, pad_id=0)

        vocab_size = len(self.vocab.id_to_string)

        self.model = LM(vocab_size=vocab_size,
                        emb_dim=args.rnn_size,
                        hidden_dim=args.rnn_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        self.model = self.model.to(args.device)

        if args.init:
            self.model.load_state_dict(checkpoint['model'])

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        if args.init:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

        self.log_interval = args.log_interval
        self.prompts = [
            "a"
        ]

    def state_dict(self):
        return {
            'vocab': self.vocab.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def prepare_prompt(self, prompt):
        device = next(self.model.parameters()).device

        prompt_list = [self.vocab.string_to_id[char] for char in prompt]
        x = torch.tensor(prompt_list).to(device).unsqueeze(1).long()

        return x, self.model.init_hidden(1)

    @torch.inference_mode()
    def complete(self, prompt, steps=512, sample=False):
        model = self.model
        model.eval()

        joiner = ''
        def cast(s):
            if isinstance(s, int):
                nonlocal joiner
                joiner = b''
                return s.to_bytes(1, 'big')
            return s

        out_list = []
        x, states = self.prepare_prompt(prompt)

        logits, states = model(x, states)
        prompt_logits = nn.functional.cross_entropy(logits[:-1], x[1:].view(-1), reduction='none').sum() / len(x[1:])
        probs = F.softmax(logits, dim=-1)
        probs = probs[-1]

        if sample:
            ix = probs.multinomial(num_samples=1)
        else:
            max_p, ix = torch.topk(probs, k=1, dim=-1)

        out_list.append(cast(self.vocab.id_to_string[int(ix)]))
        x = ix.unsqueeze(1)

        # decode
        for k in range(steps):
            logits, states = model(x, states)
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = probs.multinomial(1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            out_list.append(cast(self.vocab.id_to_string[int(ix)]))
            x = ix
        return prompt_logits, joiner.join(out_list)

    def train_one_epoch(self, epoch=0):
        model, batches = self.model, self.batches
        optimizer, loss_fn = self.optimizer, self.loss

        state = model.init_hidden(self.args.batch_size)
        model.train()

        for step in range(len(batches)):
            batch = batches[step].to(self.args.device).long()
            model.train()
            optimizer.zero_grad()
            state = model.truncate_hidden(state)

            input = batch[:-1]
            target = batch[1:].reshape(-1)

            batch_size = input.shape[1]
            prev_batch_size = state[0].shape[1]
            if batch_size != prev_batch_size:
                h,c = state
                state = h[:, :batch_size, :], c[:, :batch_size, :]
            output, state = model(input, state)
            loss = loss_fn(output, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if step % self.log_interval == 0:
                print(f"epoch {epoch} step {step}/{len(batches)} train loss: {loss.item():.3f} ppl: {loss.exp().item():.3f} grad_norm: {grad_norm.item():.3f}")
                for prompt in self.prompts:
                    _, generated_text = self.complete(prompt, 128, sample=False)
                    print('greedy', prompt + generated_text)
                for prompt in self.prompts:
                    _, generated_text = self.complete(prompt, 128, sample=True)
                    print('sample', prompt + generated_text)
                wandb.log({'train/loss': loss.item(),
                          'train/ppl': loss.exp().item(),
                          'train/lr': self.args.lr,
                          'train/epoch': epoch,
                          'train/grad_norm': grad_norm.item()})
            else:
                wandb.log({'train/loss': loss.item(),
                           'train/ppl': loss.exp().item(),
                           #'train/lr': scheduler.get_last_lr()[0],
                           'train/lr': self.args.lr,
                           'train/epoch': epoch,
                           'train/grad_norm': grad_norm.item()})


def main():
    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=Formatter)
    parser.add_argument('--init', type=Path, help="Path to checkpoint to initialize from")
    parser.add_argument('--save', type=Path, default='rnnlm.pt', help="Path to save checkpoint to")
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    parser.add_argument('--lr', default=0.0002, type=float, help='Adam learning rate')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    parser.add_argument('--epochs', default=1, type=int, help='number of training set iterations')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--bptt-len', default=256, type=int, help='RNN window size')
    parser.add_argument('--rnn-size', default=512, type=int, help='RNN width')
    parser.add_argument('--num-layers', default=1, type=int, help='RNN depth')
    parser.add_argument('--bytes-as-tokens', action='store_true', help='use bytes as tokens')
    parser.add_argument('--train', type=Path, help='Train model on this data')
    parser.add_argument('--complete', type=str, help='complete this prompt')
    parser.add_argument('--log-interval', type=int, default=100, help="Number of batches between printing training status")
    args = parser.parse_args()

    if not args.train and not args.complete:
        parser.print_help()
        print("\nPlease specify either --train or --init", file=sys.stderr)
        sys.exit(1)

    print(args)

    torch.manual_seed(3407)

    self = System(args)

    if args.train:
        wandb.init(project='rnnlm', config=args)

        for epoch in range(args.epochs):
            self.train_one_epoch(epoch=epoch)
            torch.save(self.state_dict(), args.save)

    if args.complete:
        self.model.eval()
        prompt_score, completion = self.complete("\n" + args.complete, 128, sample=False)
        print(prompt_score.item(), args.complete + completion)


if __name__ == '__main__':
    main()
