# modified from https://github.com/proger/uk4b/blob/main/model.py
# which is in turn based on https://github.com/karpathy/nanoGPT/blob/master/model.py

from dataclasses import dataclass
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lora

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

    def forward(self, x, past=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if past is not None:
            k_cache, v_cache = past
            k, v = torch.cat([k_cache, k], dim=2), torch.cat([v_cache, v], dim=-2)

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
        return y, att_entropy, torch.stack([k, v])


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

    def forward(self, x, past=None):
        x_attn, att_entropy, present = self.attn(self.ln_1(x), past=past)
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x, att_entropy, present


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

    def forward(self,
                input_ids, # (B, T)
                past=None # (nlayers, 2, B, nh, T, hs)
                ):
        device = input_ids.device
        b, t = input_ids.size()
        if past is None:
            t0 = 0
            past = torch.zeros(self.config.n_layer, 2, b, self.config.n_head, t0, self.config.n_embd // self.config.n_head, device=device)
        else:
            t0 = past.size(-2)
            t = t0 + t
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(t0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        present = past.new_empty((self.config.n_layer, 2, b, self.config.n_head, t, self.config.n_embd // self.config.n_head))
        for i, block in enumerate(self.transformer.h):
            x, _att_entropy, present[i] = block(x, past=past[i])
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits, present


@torch.inference_mode()
def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, stop_token=50256):
    """
    Take a conditioning sequence of indices input_ids (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    past = None
    for _ in range(max_new_tokens):
        if input_ids.size(1) >= self.config.block_size:
            # kv cache becomes useless here, we stop using and updating it
            past = None
            # if the past context is growing too long we must crop it at block_size
            input_ids_cond = input_ids[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(input_ids_cond, past=None)
        elif past is None:
            # forward the condition for the first time and warm up the cache
            logits, past = self(input_ids, past=None)
        else:
            # forward the last token in the sequence along with the cache
            logits, past = self(input_ids[:, [-1]], past=past)

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

        yield input_ids_next


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


def load_model(ckpt_path, *, map_location):
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    if not 'vocab_size' in checkpoint['model_args']:
        # assume checkpoint for a large model

        checkpoint['model_args']['stable_embedding'] = True
        checkpoint['model_args']['vocab_size'] = 50257
        checkpoint['model_args']['bias'] = True

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'], strict=False)
    elif '_orig_mod.transformer.h.0.attn.c_attn.lora_A.weight' in checkpoint['model']:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        lora.attach_to_c_attn(model)
        model.load_state_dict(checkpoint['model'])
    else:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'])

    model.eval()
    model.to(map_location)
    model = model._orig_mod

    return model


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
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--spm', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('ckpt_path')
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    model = load_model(args.ckpt_path, map_location=device)

    sp = spm.SentencePieceProcessor(model_file=args.spm)

    import time

    while True:
        prompt = Prompt.ask('prompt>-')
        start = [50256] + sp.encode(prompt)
        x = (torch.tensor(start, dtype=torch.long, device=device)[None, ...])

        t0 = time.time()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i, token_id in enumerate(generate(model, x, args.steps, temperature=args.temperature, top_k=args.top_k)):
                token_id = token_id.item()
                piece = sp.id_to_piece(token_id)
                if piece.startswith('▁'):
                    print(' ', end='')
                    piece = piece[1:]
                print(piece, end='', flush=True)

        t1 = time.time()
        print(f' ({i+1} tokens in {t1-t0:.2f}s)', file=sys.stderr)


if __name__ == '__main__':
    main()
