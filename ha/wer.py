import sys
from pathlib import Path

import pandas as pd
from kaldialign import align

from . import argparse


def clean_tokens(text):
    return ' '.join([token for token in text.split() if token != '␣'])


def clean_and_join_tokens(text):
    return ''.join([token for token in text.split() if token != '␣']).replace('▁', ' ')


def read_text(filename: Path):
    def clean(i, key, text):
        return i, key, clean_tokens(text)

    with open(filename) as f:
        return pd.DataFrame(
            [clean(i, *line.strip().split(maxsplit=1)) for i, line in enumerate(f)],
            columns=['dataset_index', 'media_filename', 'text']
        ).set_index('dataset_index')

def compute_alignment(hyp, ref):
    tags = []
    ins, del_, sub = 0, 0, 0
    for h, r in align(ref, hyp, '␣'):
        match h, r:
            case '␣', _:
                tags.append('+')
                ins += 1
            case _, '␣':
                tags.append('-')
                del_ += 1
            case _, _:
                if h == r:
                    tags.append('.')
                else:
                    tags.append('X')
                    sub += 1
    return {
        'tags': ''.join(tags),
        'ins': ins,
        'del': del_,
        'sub': sub,
        'total': ins + del_ + sub,
        'hyp_length': len(hyp)
    }


def compute_wer_pointwise(ref_df, hyp_df, join_bpe=False):
    clean = clean_and_join_tokens if join_bpe else clean_tokens

    wer_df = ref_df.merge(hyp_df, on='media_filename', suffixes=('_ref', '_hyp'))
    lengths = pd.DataFrame(wer_df.apply(lambda x: {'ref_length': len(clean(x['text_ref']).split())}, axis=1, result_type='expand'))
    wer_df = wer_df.join(lengths)
    edits = wer_df.apply(lambda x: compute_alignment(clean(x['text_hyp']).split(), clean(x['text_ref']).split()), axis=1, result_type='expand')
    wer_df = wer_df.join(edits)

    return wer_df


def format_wer(wer_df, tag='WER'):
    total = wer_df['total'].sum()
    ref_length = wer_df['ref_length'].sum()
    ins = wer_df['ins'].sum()
    del_ = wer_df['del'].sum()
    sub = wer_df['sub'].sum()    
    return f'%{tag}', round(100 * total / ref_length, 2), f'errors={total}/{ref_length}', f'ins={ins}', f'del={del_}', f'sub={sub}'


def main():
    parser = argparse.ArgumentParser(description='haw compares word errors', formatter_class=argparse.Formatter)
    parser.add_argument('-w', '--words', action='store_true', help='Compute WER in words by joining BPE tokens')
    parser.add_argument('ref', type=Path, help='ref')
    parser.add_argument('hyp', type=Path, help='hyp')
    args = parser.parse_args()

    ref_df = read_text(args.ref)
    hyp_df = read_text(args.hyp)
    wer_df = compute_wer_pointwise(ref_df, hyp_df, join_bpe=args.words)
    wer_df.to_csv(sys.stdout, sep='\t', index=False)
    print(*format_wer(wer_df), file=sys.stderr)


if __name__ == '__main__':
    main()
