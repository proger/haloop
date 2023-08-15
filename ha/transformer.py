from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .recognizer import Decodable, TemporalClassifier
from .attention import LayerNorm
from .conv import ConvEncoder
from .sinusoids import sinusoids_like

BlockKVCache = namedtuple('BlockKVCache', ['memory', 'time'])
Stats = namedtuple('Stats', ['meme_entropy', 'self_entropy'])


def rotate_interleaved(x, *, t0=0, base=10000):
    "rotate query or key embedding as in https://arxiv.org/abs/2104.09864 GPT-J style"
    *_, T, C = x.shape

    t = torch.arange(t0, t0+T, dtype=torch.float32, device=x.device)[:, None]
    exp = torch.arange(0, C//2, dtype=torch.float32, device=x.device)[None, :]
    exp = -2 * exp.repeat_interleave(2, -1) / C

    sin = torch.sin((base**exp) * t)
    cos = torch.cos((base**exp) * t)

    odd, even = x[..., 0::2], x[..., 1::2]
    x_ = torch.stack([-even, odd], dim=-1).flatten(-2, -1)

    x = x * cos + x_ * sin
    return x


class CTCAttentionDecoder(nn.Module, Decodable):
    "CTC loss on the encoder, CE loss on the decoder"
    def __init__(self, *, vocab: int, head_dim: int, heads: int, p_drop: float, layers: int):
        super().__init__()
        self.decoder = Decoder(vocab=vocab, head_dim=head_dim, heads=heads, p_drop=p_drop, layers=layers)
        self.recognizer = TemporalClassifier(feat_dim=head_dim * heads, vocab_size=vocab)

    def forward(
        self,
        features, targets, input_lengths=None, target_lengths=None,
        star_penalty=None,
        measure_entropy=False,
        drop_labels=False,
    ):
        decoder_loss, decoder_stats = self.decoder(features, targets, input_lengths, target_lengths, star_penalty, measure_entropy, drop_labels)
        recognizer_loss, recognizer_stats = self.recognizer(features, targets, input_lengths, target_lengths, star_penalty)
        return decoder_loss + 0.3*recognizer_loss, {**decoder_stats, **recognizer_stats}

    def decode(self, features, input_lengths, target_lengths):
        return self.decoder.decode(features, input_lengths, target_lengths)


class Decoder(nn.Module, Decodable):
    def __init__(self, *, vocab: int, head_dim: int, heads: int, p_drop: float, layers: int):
        super().__init__()

        self.wte = nn.Embedding(vocab, head_dim * heads)

        self.h = nn.ModuleList([
            Block(head_dim=head_dim, heads=heads, p_drop=p_drop, memory=True)
            for _ in range(layers)
        ])
        self.ln_f = LayerNorm(head_dim * heads, bias=False)
        self.lm_head = nn.Linear(head_dim * heads, vocab, bias=False)

    def forward(
        self,
        features, targets, input_lengths=None, target_lengths=None,
        star_penalty=None, # ignored
        measure_entropy=False,
        drop_labels=None,
    ):
        N, T = targets.shape

        # make a training prompt:
        # add <s> token to the beginning of each target sequence
        # prompt: STX a b c
        # target: a b c ETX

        # prompt1: STX a b c
        # prompt2: STX a b c d
        # target1: a b c ETX PAD
        # target2: a b c d ETX
        stx, etx = 2, 3
        prompt = F.pad(targets, (1, 0), value=stx) # (N, T+1)
        targets = F.pad(targets, (0, 1), value=0) # (N, T+1)
        targets[torch.arange(N, device=targets.device), target_lengths] = torch.LongTensor([etx]).to(targets.device)
        T = T + 1

        stats = Stats(meme_entropy=[], self_entropy=[])

        if (drop_labels is None and self.training) or drop_labels:
            # label dropout
            keep = torch.empty_like(prompt).bernoulli_(0.9).bool()
            prompt = torch.where(keep, prompt, torch.ones_like(prompt))

        # run all tokens at once
        y = self.wte(prompt)
        for block in self.h:
            y, (m_ent, t_ent), _ = block(
                y,
                causal=True, memory=features, memory_lengths=input_lengths,
                measure_entropy=measure_entropy
            )
            stats.meme_entropy.append(m_ent)
            stats.self_entropy.append(t_ent)

        logits = self.lm_head(self.ln_f(y)) # (N, T, V)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='mean')
        return loss, stats._asdict()

    def decode(self, features, input_lengths, target_lengths):
        "Perform batched greedy decoding."

        N, S, _C = features.shape
        T = target_lengths.max().item()+1

        # make an inference prompt:
        # add <s> token to the beginning of each target sequence
        stx, etx = 2, 3 # <s>/BOS/␂ token
        prompt = input_lengths.new_zeros((N, T+1)) + etx
        prompt[:, 0] = stx
        output_lengths = input_lengths.new_zeros((N,))

        L = len(self.h)
        heads, head_dim = self.h[0].heads, self.h[0].head_dim # assume all layers are the same
        kv_cache = BlockKVCache(
            memory=features.new_zeros((L, 2, N, heads, S, head_dim), dtype=torch.float16),
            time=features.new_zeros((L, 2, N, heads, T, head_dim), dtype=torch.float16),
        )
        memory_empty = torch.ones((N,), dtype=torch.bool, device=features.device)

        alive = prompt.new_ones((N,), dtype=torch.bool)

        for t in range(T):
            # run one token at a time
            input = prompt[alive, None, [t]]
            #print('input', input.shape, input)
            if input.shape[0] == 0:
                break
            y = self.wte(input)

            for layer, block in enumerate(self.h):
                y, _, new_cache = block(
                    y,
                    causal=True, memory=features[alive], memory_lengths=input_lengths[alive],
                    measure_entropy=False,
                    kv_cache_parts=BlockKVCache(
                        memory=(kv_cache.memory, alive, memory_empty, layer),
                        time=(kv_cache.time, alive, None, layer)
                    ),
                    t0=t,
                )
                assert kv_cache.memory[0].untyped_storage().data_ptr()  == new_cache.memory[0].untyped_storage().data_ptr()

            logits = self.lm_head(self.ln_f(y[:, -1, :])) # (N, V)

            output_lengths[alive] += 1
            token = logits.argmax(dim=-1)
            prompt[alive, t+1] = token.int()
            alive_ = alive.clone()
            alive[alive_] = alive[alive_] & (token != etx)

        outputs = torch.nested.nested_tensor([p[1:l] for p, l in zip(prompt, output_lengths)])
        alignments = [None]*N
        return outputs, alignments
        

class AudioEncoder(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int = 64,
        heads: int = 12,
        p_drop: float = 0.2,
        layers: int = 12,
        input_dim: int = 80,
        conv_dim: int = 256,
        conv_strides: tuple[int, ...] = (2,2,2)
    ):
        super().__init__()

        self.head_dim = head_dim
        self.heads = heads

        self.conv = ConvEncoder(
            input_dim=input_dim,
            hidden_dim=conv_dim,
            output_dim=head_dim*heads,
            strides=conv_strides
        )
        self.drop = nn.Dropout(p_drop)
        self.h = nn.ModuleList([
            Block(head_dim=head_dim, heads=heads, p_drop=p_drop)
            for _ in range(layers)
        ])
        self.ln_f = LayerNorm(head_dim * heads, bias=False)

    def subsampled_lengths(self, input_lengths):
        return self.conv.subsampled_lengths(input_lengths)

    def forward(self, x, input_lengths, measure_entropy=False):
        x = x.mT
        x, input_lengths = self.conv(x, input_lengths)
        x = x.mT

        x = self.drop(x)

        stats = Stats(meme_entropy=[], self_entropy=[])

        time_mask = torch.arange(x.size(-2), device=x.device)[None, ...] >= input_lengths[..., None]
        time_mask = time_mask[:, None, None, :]
        #time_mask = None # ????

        for block in self.h:
            x, (m_ent, t_ent), _ = block(
                x,
                time_mask=time_mask,
                measure_entropy=measure_entropy
            )
            stats.meme_entropy.append(m_ent)
            stats.self_entropy.append(t_ent)

        x = self.ln_f(x)
        return x, input_lengths, stats._asdict()


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

    def init_from_flash_mha_(self, mha):
        #from flash_attn.modules.mha import MHA
        step = self.head_dim * self.heads
        assert mha.Wqkv.weight.shape[0] == step * 3
        self.q.weight.data = mha.Wqkv.weight.data[0*step:1*step, :]
        self.k.weight.data = mha.Wqkv.weight.data[1*step:2*step, :]
        self.v.weight.data = mha.Wqkv.weight.data[2*step:3*step, :]
        self.proj.weight.data = mha.out_proj.weight.data
        return self

    def read_memory(self, memory):
        N, S, _C = memory.shape
        heads, head_dim = self.heads, self.head_dim
        k = self.k(memory) # (N, S, head_dim * heads)
        v = self.v(memory) # (N, S, head_dim * heads)
        k = k.view(N, S, heads, head_dim).transpose(-3, -2) # (N, heads, S, head_dim)
        v = v.view(N, S, heads, head_dim).transpose(-3, -2) # (N, heads, S, head_dim)
        return k, v

    def forward(
        self,
        x,
        memory,
        *,
        mask=None, # (N, ..., T, S) | None
        causal=False,
        measure_entropy=False,
        kv_cache_parts=None, # ((L, 2, N+M, heads, S, head_dim), N+M, L, layer)
        t0=0,
        rope=False,
    ):
        N, T, _C = x.shape
        N, S, _C = memory.shape
        heads, head_dim = self.heads, self.head_dim

        q = self.q(x) # (N, T, head_dim * heads)
        q = q.view(N, T, heads, head_dim).transpose(-3, -2) # (N, heads, T, head_dim)

        if causal:
            # Causal self-attention: add new memory to kv cache
            k, v = self.read_memory(memory)

            if kv_cache_parts is not None:
                kv_cache, alive, empty, layer = kv_cache_parts
                kv_cache[layer, 0, alive, :, t0:t0+S, :] = k
                kv_cache[layer, 1, alive, :, t0:t0+S, :] = v

                k = kv_cache[layer, 0, alive, :, :t0+S, :] # (N, heads, t0+S, head_dim)
                v = kv_cache[layer, 1, alive, :, :t0+S, :] # (N, heads, t0+S, head_dim)

                # Query token attends to everything in the cache now.
                causal = False
                assert T == 1, "Causal self-attention with caching supports only one token at a time."
        elif kv_cache_parts is not None:
            # Cross attention: warm up the cache once.
            kv_cache, alive, empty, layer = kv_cache_parts
            if empty[layer]:
                # Warm up the cache once.
                k, v = self.read_memory(memory)

                kv_cache[layer, 0, alive, :, :, :] = k
                kv_cache[layer, 1, alive, :, :, :] = v
                empty[layer] = False
            else:
                k = kv_cache[layer, 0, alive, :, :, :] # (N, heads, S, head_dim)
                v = kv_cache[layer, 1, alive, :, :, :] # (N, heads, S, head_dim)
        else:
            # Bidirectional self-attention: no use for the cache
            k, v = self.read_memory(memory)

        if rope:
            q = rotate_interleaved(q, t0=t0).to(q.dtype)
            k = rotate_interleaved(k).to(k.dtype)

        if not measure_entropy:
            if mask is not None:
                mask = ~mask
                is_causal = False
            else:
                is_causal = causal

            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=self.p_drop if self.training else 0)

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
        x = x.transpose(-3, -2).reshape(N, T, heads * head_dim) # (N, T, heads * head_dim)
        
        return self.dropout(self.proj(x)), att_entropy, kv_cache_parts


class Block(nn.Module):
    def __init__(self, head_dim: int, heads: int, p_drop: float, memory=False):
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim

        self.ln_time = LayerNorm(head_dim * heads, bias=False)
        if True:
            self.mix_time = MultiHeadAttention(head_dim=head_dim, heads=heads, p_drop=p_drop)
        else:
            from flash_attn.modules.mha import MHA
            self.mix_time = MHA(
                embed_dim=head_dim * heads,
                num_heads=heads,
                cross_attn=False,
                qkv_proj_bias=False,
                out_proj_bias=False,
                dropout=p_drop,
                causal=memory,
                rotary_emb_dim=head_dim,
                use_flash_attn=True,
                layer_idx=-1,
            )
        if memory:
            self.mix_memory = MultiHeadAttention(head_dim=head_dim, heads=heads, p_drop=p_drop)
        else:
            self.mix_memory = None
        self.ln_chan = LayerNorm(head_dim * heads, bias=False)
        self.mix_chan = nn.Sequential(
            nn.Linear(head_dim * heads, head_dim * heads * 4, bias=False),
            nn.GELU(),
            nn.Linear(head_dim * heads * 4, head_dim * heads, bias=False),
            nn.Dropout(p_drop),
        )

    def forward(
        self, x, time_mask=None, causal=False,
        memory=None, memory_lengths=None,
        measure_entropy=False,
        kv_cache_parts=BlockKVCache(memory=None, time=None),
        t0=0,
    ):
        x_norm = self.ln_time(x)

        if self.mix_memory is not None:
            memory_mask = torch.arange(memory.shape[-2], device=x.device)[None, :] >= memory_lengths[:, None]
            m, m_ent, memory_cache = self.mix_memory(
                x_norm,
                memory,
                mask=memory_mask[:, None, None, :],
                measure_entropy=measure_entropy,
                kv_cache_parts=kv_cache_parts.memory,
                t0=t0,
            )
            x = x + m
        else:
            m_ent = torch.tensor(float('-inf'))
            memory_cache = None

        t, t_ent, time_cache = self.mix_time(x_norm, x_norm, mask=time_mask, causal=causal, measure_entropy=measure_entropy, kv_cache_parts=kv_cache_parts.time, t0=t0, rope=True)

        x = x + t
        x = x + self.mix_chan(self.ln_chan(x))
        return x, (m_ent, t_ent), BlockKVCache(memory=memory_cache, time=time_cache)
