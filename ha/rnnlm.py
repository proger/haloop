from pathlib import Path
import argparse

from rich.console import Console
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .symbol_tape import Vocabulary, tokenize_bytes, tokenize_chars, SymbolTape

console = Console()
def print(*args, flush=False, **kwargs):
    console.log(*args, **kwargs)


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

    def forward(self,
                input, # (T, N)
                state # ((L, N, H), (L, N, H))
                ):
        emb = self.embedding(input)

        output, state = self.rnn(emb, state)
        output = self.out_layer(output) # (T, N, V)

        output = output.view(-1, self.num_classes)
        return output, state

    def forward_batch_first(self,
                            input, # (N, T),
                            state # ((L, N, H), (L, N, H))
                            ):
        emb = self.embedding(input)
        emb = emb.transpose(0,1) # (T, N, H)

        output, state = self.rnn(emb, state)
        output = self.out_layer(output) # (T, N, V)
        output = output.transpose(0,1) # (N, T, V)

        return output, state

    def init_hidden(self, batch_size):
        weight = self.out_layer.weight
        h = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        c = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h,c)

    def truncate_hidden(self, state):
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
            if args.vocab_from_data:
                self.data, self.vocab = tokenize_chars(args.train, self.vocab, extend_vocab=extend_vocab)
            else:
                self.data, self.vocab = tokenize_bytes(args.train, self.vocab, extend_vocab=extend_vocab)
            self.dataset = SymbolTape(self.data, args.batch_size, args.bptt_len, pad_id=0)

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

        self.model = LM(vocab_size=vocab_size,
                        emb_dim=args.rnn_size,
                        hidden_dim=args.rnn_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        self.model = self.model.to(args.device)

        if args.init:
            self.model.load_state_dict(checkpoint['model'])

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=args.lr, weight_decay=0.01)
        if args.init:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.scaler = torch.cuda.amp.GradScaler()
        if args.init:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

        self.log_interval = args.log_interval
        if args.vocab_from_data:
            self.prompts = args.complete
        else:
            self.prompts = [prompt.encode('utf-8') for prompt in args.complete]
        self.args = args

    def state_dict(self):
        return {
            'args': vars(self.args),
            'vocab': self.vocab.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }

    def prepare_prompt(self, prompt):
        device = next(self.model.parameters()).device

        prompt_list = [self.vocab.string_to_id[char] if isinstance(char, str) else char for char in prompt]
        x = torch.tensor(prompt_list).to(device).unsqueeze(1).long()

        return x, self.model.init_hidden(1)

    @torch.inference_mode()
    def complete(self, prompt, steps=512, top_k=1):
        model = self.model
        model.eval()

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

        out_list = []
        x, states = self.prepare_prompt(prompt)

        logits, states = model(x, states)
        prompt_logits = nn.functional.cross_entropy(logits[:-1], x[1:].view(-1), reduction='none').sum() / len(x[1:])

        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        probs = probs[-1]

        ix = probs.multinomial(num_samples=1)

        out_list.append(cast(self.vocab.id_to_string[int(ix)]))
        x = ix.unsqueeze(1)

        # decode
        for k in range(steps):
            logits, states = model(x, states)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            ix = probs.multinomial(num_samples=1)
            out_list.append(cast(self.vocab.id_to_string[int(ix)]))
            x = ix
        return prompt_logits, joiner.join(out_list)

    def train_one_epoch(self, epoch=0, step=0):
        model, batches = self.model, self.batches
        optimizer, scaler, loss_fn = self.optimizer, self.scaler, self.loss

        state = model.init_hidden(self.args.batch_size)
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(batches, start=step):
            batch = batch.to(self.args.device).long().squeeze(0)
            state = model.truncate_hidden(state)

            input = batch[:-1]
            target = batch[1:].reshape(-1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output, state = model(input, state)
                loss = loss_fn(output, target)

            if torch.isnan(loss):
                print(f'[{epoch + 1}, {i + 1:5d}], loss is nan, skipping batch', flush=True)
                scaler.update()
                continue

            if torch.isinf(loss):
                print(f'[{epoch + 1}, {i + 1:5d}], loss is inf, skipping batch, skipping scaler update', flush=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                print(f'[{epoch + 1}, {i + 1:5d}], grad_norm is inf or nan, skipping batch', flush=True)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if i % self.log_interval == 0:
                _, outputs = self.evaluate()
                print(f"epoch {epoch} step {i}/{len(batches)} loss: {loss.item():.3f} ppl: {loss.exp().item():.3f} grad_norm: {grad_norm.item():.3f} {'; '.join(outputs)}")
                wandb.log({'train/loss': loss.item(),
                           'train/ppl': loss.exp().item(),
                           'train/lr': self.args.lr,
                           'train/epoch': epoch,
                           'train/grad_norm': grad_norm.item()})
                model.train()
            else:
                wandb.log({'train/loss': loss.item(),
                           'train/ppl': loss.exp().item(),
                           #'train/lr': scheduler.get_last_lr()[0],
                           'train/lr': self.args.lr,
                           'train/epoch': epoch,
                           'train/grad_norm': grad_norm.item()})

            if self.args.max_steps >= 0 and i == self.args.max_steps:
                break

        return i+1

    def evaluate(self):
        prompt_scores = []
        outputs = []
        for prompt in self.prompts:
            prompt_score, completion = self.complete(prompt, self.args.bptt_len, top_k=self.args.top_k)
            if isinstance(completion, bytes):
                outputs.append(str(prompt + completion, 'utf-8', errors='replace'))
            else:
                outputs.append(prompt + completion)
            prompt_scores.append(prompt_score.item())
        return prompt_scores, outputs


def main():
    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description="hal trains recurrent language models", formatter_class=Formatter)
    parser.add_argument('--init', type=Path, help="Path to checkpoint to initialize from")
    parser.add_argument('--save', type=Path, default='rnnlm.pt', help="Path to save checkpoint to")
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    parser.add_argument('--lr', default=0.002, type=float, help='Adam learning rate')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    parser.add_argument('--epochs', default=1, type=int, help='number of training set iterations')
    parser.add_argument('--max-steps', default=-1, type=int, help='maximum number of training steps per epoch (useful for e.g. lr search)')
    parser.add_argument('--batch-size', default=1280, type=int, help='batch size')
    parser.add_argument('--bptt-len', default=256, type=int, help='RNN window size')
    parser.add_argument('--rnn-size', default=2048, type=int, help='RNN width')
    parser.add_argument('--num-layers', default=1, type=int, help='RNN depth')
    parser.add_argument('--vocab-from-data', action='store_true', help='build character vocabulary from the data')
    parser.add_argument('--train', type=Path, help='Train model on this data')
    parser.add_argument('--top-k', type=int, default=1, help='top-k sampling')
    parser.add_argument('--log-interval', type=int, default=100, help="Number of batches between printing training status")
    #parser.add_argument('--complete, type=str, nargs='+', default=['\nа', '\nя'], help="Prompts to complete during evaluation")
    parser.add_argument('--complete', type=str, nargs='+', default=['\nhello', '\nwhat '], help="Prompts to complete during evaluation")
    parser.add_argument('--num-workers', type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args()

    torch.manual_seed(3407)

    self = System(args)

    if args.train:
        print(args)
        wandb.init(project='rnnlm', config=args)

        step = 0
        try:
            for epoch in range(args.epochs):
                step = self.train_one_epoch(epoch=epoch, step=step)
                torch.save(self.state_dict(), args.save)
        except KeyboardInterrupt:
            print('interrupted, saving')
            torch.save(self.state_dict(), args.save)

    for prompt_score, completion in zip(*self.evaluate()):
        print('{:.2f}'.format(prompt_score), completion)

if __name__ == '__main__':
    main()
