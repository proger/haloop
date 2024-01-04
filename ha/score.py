import argparse
from itertools import islice
import sys

import torch
import torch.nn as nn

try:
    import sentencepiece as spm
except ImportError:
    print("Please install sentencepiece with: pip install sentencepiece", file=sys.stderr)
    raise

from .init import load_model


# https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Score sentences with GPT. Prints three columns: negative log likelihood per token, number of tokens in the prompt and total number of tokens in the input before truncation.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--spm', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('ckpt_path')
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    model = load_model(args.ckpt_path, map_location=device)
    assert model.config.causal
    if args.compile:
        model = torch.compile(model)
    print('Loaded model:', model.config, file=sys.stderr)

    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    sp = spm.SentencePieceProcessor(model_file=args.spm)
    class Tok:
        eos = 50256

    for i, lines in enumerate(batched(sys.stdin, args.batch_size)):
        completion_tokens = sp.encode([p.strip() for p in lines])
        completions = nn.utils.rnn.pad_sequence(
            [torch.LongTensor(p) for p in completion_tokens],
            batch_first=True,
            padding_value=0
        ).to(device=device)

        if completions.size(-1) >= model.config.block_size:
            print(f'warning: batch {i} is too wide (shape {completions.shape}) and will be truncated', file=sys.stderr)
            completions = completions[:, :model.config.block_size].contiguous()

        prompts = torch.full((len(completions), 1), Tok.eos, dtype=torch.long, device=device)
        input_ids = torch.cat([prompts, completions[..., :-1]], dim=-1)[:, :model.config.block_size]

        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            logits = model.forward_all(input_ids=input_ids, target_ids=completions, reduction='none')
            logits = logits.view(-1, input_ids.shape[-1])
            for loss, tokens in zip(logits.sum(-1), completion_tokens):
                num_tokens = min(model.config.block_size, len(tokens))
                loss_per_token = loss.item() / num_tokens
                print(f'{loss_per_token:0.3f}', num_tokens, len(tokens), sep='\t', flush=True)


if __name__ == '__main__':
    main()
