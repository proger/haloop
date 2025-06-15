import argparse
import numpy as np
import sentencepiece as spm
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='bpe model filename')
parser.add_argument('--block', type=int, help='one document per line, padded up to this many tokens')
parser.add_argument('input_txt', type=str)
parser.add_argument('output_bin', type=str)
args = parser.parse_args()


if args.model:
    sp = spm.SentencePieceProcessor(model_file=args.model)

if args.block:
    lines = ['\n' + line.strip() + '\n' for line in open(args.input_txt)]
    byte_lines = [line.encode('utf-8') for line in lines]
    if args.model:
        ids = [sp.encode(line) for line in lines]
        max_len = max(max(map(len, ids)), args.block)
        ids = [line + [0] * (max_len - len(line)) for line in ids]
    else:
        max_len = max(max(len(line) for line in byte_lines), args.block)
        ids = [list(line + b'\0' * (max_len - len(line))) for line in byte_lines]

    real_bytes = sum([len(line) - 1 for line in byte_lines]) # off by one for shifting
    assert max_len == args.block, f"some lines are too long: found {max_len=}"
    ids = list(reduce(list.__add__, ids))
    assert len(ids) % args.block == 0, f"{len(ids) % args.block=}"
else:
    chars = open(args.input_txt).read()
    bytes = chars.encode('utf-8')
    if args.model:
        ids = sp.encode(chars)
    else:
        ids = list(bytes)
    real_bytes = len(bytes)

arr = np.memmap(args.output_bin, dtype=np.uint16, mode='w+', shape=(len(ids),))
arr[:] = ids
arr.flush()
print("wrote", len(ids), "tokens to", args.output_bin, "real bytes", real_bytes)
