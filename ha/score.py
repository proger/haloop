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
    parser = argparse.ArgumentParser(description='Score sentences with GPT. Prints two columns: negative log likelihood per token and number of tokens in the prompt.')
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
    print('Loaded model:', model.config, file=sys.stderr)
    assert model.config.causal
    if args.compile:
        model = torch.compile(model)

    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    sp = spm.SentencePieceProcessor(model_file=args.spm)
    class Tok:
        eos = 50256

    for lines in batched(sys.stdin, args.batch_size):
        completion_tokens = sp.encode([p.strip() for p in lines])
        completions = nn.utils.rnn.pad_sequence(
            [torch.LongTensor(p) for p in completion_tokens],
            batch_first=True,
            padding_value=0
        ).to(device=device)
        prompts = torch.full((len(completions), 1), Tok.eos, dtype=torch.long, device=device)
        input_ids = torch.cat([prompts, completions[..., :-1]], dim=-1)

        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            logits = model.forward_all(input_ids=input_ids, target_ids=completions, reduction='none')
            logits = logits.view(-1, input_ids.shape[-1])
            for loss, tokens in zip(logits.sum(-1), completion_tokens):
                loss_per_token = loss.item() / len(tokens)
                print(f'{loss_per_token:0.3f}', len(tokens), sep='\t')


if __name__ == '__main__':
    main()
