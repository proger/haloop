# modified from https://github.com/proger/uk4b/blob/main/model.py
# which is in turn based on https://github.com/karpathy/nanoGPT/blob/master/model.py

from dataclasses import dataclass
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MonitoredCausalSelfAttention(nn.Module):
    """CausalSelfAttention that measures attention entropy"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        bias = torch.tril(x.new_ones(T, T)).view(1, 1, T, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(bias == 0, float('-inf'))
        att = att.softmax(dim=-1) # (B, nh, T, T)

        # measure attention entropy
        att_entropy = (-att * torch.log(att + 1e-8)).sum(dim=-1).mean(dim=(0,1,2))

        # attend
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att_entropy


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MonitoredCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x_attn, att_entropy = self.attn(self.ln_1(x))
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.stable_embedding:
            try:
                import bitsandbytes as bnb
            except ImportError:
                print("Please install bitsandbytes with: pip install bitsandbytes", file=sys.stderr)
                raise
            embedding = bnb.nn.StableEmbedding
        else:
            embedding = nn.Embedding

        self.transformer = nn.ModuleDict(dict(
            wte = embedding(config.vocab_size, config.n_embd),
            wpe = embedding(config.block_size, config.n_embd),

            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, input_ids):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits


@torch.inference_mode()
def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, stop_token=50256):
    """
    Take a conditioning sequence of indices input_ids (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        input_ids_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = self(input_ids_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        input_ids_next = torch.multinomial(probs, num_samples=1)
        if input_ids_next == stop_token:
            # time to stop
            break
        else:
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

    return input_ids


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    stable_embedding: bool = False


@torch.inference_mode()
def main():
    import argparse
    from rich.prompt import Prompt

    try:
        import sentencepiece as spm
    except ImportError:
        print("Please install sentencepiece with: pip install sentencepiece", file=sys.stderr)
        raise

    parser = argparse.ArgumentParser('GPT REPL')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--spm', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('ckpt_path')
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    checkpoint = torch.load(args.ckpt_path, map_location=device)

    if not 'vocab_size' in checkpoint['model_args']:
        # assume checkpoint for a large model

        checkpoint['model_args']['stable_embedding'] = True
        checkpoint['model_args']['vocab_size'] = 50257
        checkpoint['model_args']['bias'] = True

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'], strict=False)
    else:        
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'])

    model.eval()
    model.to(device)
    model = model._orig_mod

    sp = spm.SentencePieceProcessor(model_file=args.spm)

    while True:
        prompt = Prompt.ask('prompt>-')
        start = [50256] + sp.encode(prompt)
        x = (torch.tensor(start, dtype=torch.long, device=device)[None, ...])

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = generate(model, x, args.steps, temperature=args.temperature, top_k=args.top_k)

        y = y[0].tolist()
        _prefix, gen = y[:len(start)], y[len(start):]
        try:
            eot = gen.index(50256)
            gen = gen[:eot]
        except ValueError:
            pass

        print(sp.decode(gen))


if __name__ == '__main__':
    main()