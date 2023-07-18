import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .recognizer import Decodable


class Decoder(nn.Module, Decodable):
    def __init__(self, *, context: int, vocab: int, head_dim: int, heads: int, p_drop: float, layers: int):
        super().__init__()

        self.wpe = nn.Embedding(context, head_dim * heads)
        self.wte = nn.Embedding(vocab, head_dim * heads)

        self.h = nn.ModuleList([
            Block(head_dim=head_dim, heads=heads, p_drop=p_drop, memory=True)
            for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(head_dim * heads)
        self.lm_head = nn.Linear(head_dim * heads, vocab, bias=False)

    def forward(
        self,
        features, targets, input_lengths=None, target_lengths=None,
        star_penalty=None, # ignored
        measure_entropy=False,
    ):
        N, T = targets.shape

        # make a training prompt:
        # add <s> token to the beginning of each target sequence
        # prompt: STX a b c
        # target: a b c ETX

        # prompt1: STX a b c
        # prompt2: STX a b c d
        # target1: a b c PAD
        # target2: a b c d
        stx, etx = 2, 3
        prompt = F.pad(targets, (1, 0), value=stx) # (N, T+1)
        targets = F.pad(targets, (0, 1), value=0) # (N, T+1)
        targets[:, target_lengths] = etx
        T = T + 1

        stats = {'meme_entropy': [], 'self_entropy': []}

        # run all tokens at once
        y = self.wte(prompt) + self.wpe(torch.arange(T, device=prompt.device))
        causal_mask = torch.triu(y.new_ones(T, T), diagonal=1).bool()
        for block in self.h:
            y, (m_ent, t_ent) = block(
                y,
                time_mask=causal_mask, memory=features, memory_lengths=input_lengths,
                measure_entropy=measure_entropy
            )
            stats['meme_entropy'].append(m_ent)
            stats['self_entropy'].append(t_ent)

        logits = self.lm_head(self.ln_f(y)) # (N, T, V)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return loss, stats

    def decode(self, features, input_lengths, target_lengths):
        N, S, _C = features.shape
        stx = 2 # <s>/BOS/␂ token
        prompt = input_lengths.new_zeros((N, T := 1)) + stx

        y = features.new_zeros((N, 0, self.wte.embedding_dim))
        for _ in range(target_lengths.max().item()):
            # run one token at a time
            y_ = self.wte(prompt[:, [-1]]) + self.wpe(torch.arange(T-1, T, device=prompt.device))

            # TODO: use kv caching
            y = torch.cat([y, y_], dim=1) # (N, T, C)
            causal_mask = torch.triu(y.new_ones(T, T), diagonal=1).bool()
            for block in self.h:
                y, _ = block(y, time_mask=causal_mask, memory=features, memory_lengths=input_lengths)
            logits = self.lm_head(self.ln_f(y[:, [-1], :])) # (N, 1, V)

            token = logits.argmax(dim=-1) # (N, 1)
            prompt = torch.cat([prompt, token], dim=-1) # (N, T+1)
            T += 1

        prompt = prompt[:, 1:] # (N, T)
        alignments = [None]*N
        return prompt, alignments
        


class MultiHeadAttention(nn.Module):
    def __init__(self, head_dim: int = 64, heads: int = 12, p_drop: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.q = nn.Linear(head_dim * heads, head_dim * heads, bias=False)
        self.k = nn.Linear(head_dim * heads, head_dim * heads, bias=False)
        self.v = nn.Linear(head_dim * heads, head_dim * heads, bias=False)
        
        self.proj = nn.Linear(head_dim * heads, head_dim * heads, bias=False)
        self.p_drop = p_drop
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, memory, mask=None, measure_entropy=False):
        N, T, _C = x.shape
        N, S, _C = memory.shape
        heads, head_dim = self.heads, self.head_dim

        q = self.q(x) # (N, T, head_dim * heads)
        k = self.k(memory) # (N, S, head_dim * heads)
        v = self.v(memory) # (N, S, head_dim * heads)

        q = q.view(N, T, heads, head_dim).transpose(1, 2) # (N, heads, T, head_dim)
        k = k.view(N, S, heads, head_dim).transpose(1, 2) # (N, heads, S, head_dim)
        v = v.view(N, S, heads, head_dim).transpose(1, 2) # (N, heads, S, head_dim)

        if not measure_entropy:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask, dropout_p=self.p_drop if self.training else 0)

            att_entropy = torch.tensor(float('-inf'))
        else:
            qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1]) # (N, heads, T, S)
            if mask is not None:
                # use ~mask with flash attention
                qk = qk.masked_fill(mask, float('-inf'))

            att = qk.softmax(dim=-1)

            # measure attention entropy
            att_entropy = (-att * torch.log(att + 1e-8)).sum(dim=-1).mean(dim=(0,1,2))

            x = att @ v # (N, heads, T, head_dim)
        x = x.transpose(1, 2).reshape(N, T, heads * head_dim) # (N, T, heads * head_dim)
        
        return self.dropout(self.proj(x)), att_entropy


class Block(nn.Module):
    def __init__(self, head_dim: int, heads: int, p_drop: float, memory=False):
        super().__init__()

        self.ln_time = nn.LayerNorm(head_dim * heads)
        self.mix_time = MultiHeadAttention(head_dim=head_dim, heads=heads, p_drop=p_drop)
        if memory:
            self.mix_memory = MultiHeadAttention(head_dim=head_dim, heads=heads, p_drop=p_drop)
        else:
            self.mix_memory = None
        self.ln_chan = nn.LayerNorm(head_dim * heads)
        self.mix_chan = nn.Sequential(
            nn.Linear(head_dim * heads, head_dim * heads * 4, bias=False),
            nn.GELU(),
            nn.Linear(head_dim * heads * 4, head_dim * heads, bias=False),
            nn.Dropout(p_drop),
        )

    def forward(self, x, time_mask=None, memory=None, memory_lengths=None, measure_entropy=False):
        x = self.ln_time(x)

        if self.mix_memory is not None:
            memory_mask = torch.arange(memory.shape[-2], device=x.device)[None, :] >= memory_lengths[:, None]
            m, m_ent = self.mix_memory(x, memory, mask=memory_mask[:, None, None, :], measure_entropy=measure_entropy)
            x = x + m
        else:
            m_ent = float('-inf')

        t, t_ent = self.mix_time(x, x, mask=time_mask, measure_entropy=measure_entropy)
        x = x + t
        x = x + self.mix_chan(self.ln_chan(x))
        return x, (m_ent, t_ent)
