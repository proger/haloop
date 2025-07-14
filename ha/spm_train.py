# see also: https://a.wilab.org.ua/gpt/spm_train.py
import argparse
import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=512, help='vocab size')
parser.add_argument('--model', type=str, default='bpe512.model', help='model name')
parser.add_argument('corpus_txt', type=str)
args = parser.parse_args()


with open(args.model, 'wb') as model_writer:
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter([open(args.corpus_txt, 'rb').read().decode('utf-8', errors='ignore')]),
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

