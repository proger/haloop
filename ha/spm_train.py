# see also: https://a.wilab.org.ua/gpt/spm_train.py
import argparse
import math
import sentencepiece as spm
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=512, help='vocab size')
parser.add_argument('--model', type=str, default='bpe512.model', help='model name')
parser.add_argument('--test', type=str, default='', help='test entropy on this corpus')
parser.add_argument('corpus_txt', type=str)
args = parser.parse_args()


def read(filename):
    return open(filename, 'rb').read().decode('utf-8', errors='ignore')

corpus = read(args.corpus_txt)

with open(args.model, 'wb') as model_writer:
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter([corpus]),
        max_sentence_length=1<<31-1,
        model_writer=model_writer,
        vocab_size=args.vocab_size,
        num_threads=32,
        character_coverage=0.9995,
        model_type='bpe',
        split_digits=True,
        #allow_whitespace_only_pieces=True,
        normalization_rule_name='nfkc',
        #normalization_rule_name='nmt_nfkc',
        byte_fallback=True,
        add_dummy_prefix=True, # encode("Hello World") = encode(" Hello World")
        control_symbols=[])

if args.test:
    sp = spm.SentencePieceProcessor()
    sp.load(args.model)
    pieces = sp.encode_as_ids(corpus)
    freq = Counter(pieces)
    freq.update([k for k in range(args.vocab_size)])  # laplace smoothing
    z = sum([freq[k] for k in freq])

    test = read(args.test)
    test_pieces = sp.encode_as_ids(test)
    bits = sum(- math.log2(freq[p]/z) for p in test_pieces)
    bytes = bits / 8
    print(f"{bits=} {bytes=} {len(test_pieces)=}", args.test)
