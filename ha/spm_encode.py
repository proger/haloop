import argparse
import numpy as np
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='bpe model filename')
parser.add_argument('input_txt', type=str)
parser.add_argument('output_bin', type=str)
args = parser.parse_args()

if args.model:
    sp = spm.SentencePieceProcessor(model_file=args.model)
    ids = sp.encode(open(args.input_txt).read())
else:
    ids = list(open(args.input_txt, 'rb').read())

arr = np.memmap(args.output_bin, dtype=np.uint16, mode='w+', shape=(len(ids),))
arr[:] = ids
arr.flush()
print("wrote", len(ids), "tokens to", args.output_bin)
