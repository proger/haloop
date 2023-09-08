import sys
from pathlib import Path

import pandas as pd
from kaldialign import align

from . import argparse


def clean_tokens(text):
    return ' '.join([token for token in text.split() if token != '␣'])


def clean_and_join_tokens(text):
    return ''.join([token for token in text.split() if token != '␣']).replace('▁', ' ')


def read_text(filename: Path, join_bpe: bool = False):
    def clean(i, key, text):
        if join_bpe:
            return i, key, clean_and_join_tokens(text)
        else:
            return i, key, clean_tokens(text)

    with open(filename) as f:
        return pd.DataFrame(
            [clean(i, *line.strip().split(maxsplit=1)) for i, line in enumerate(f)],
            columns=['dataset_index', 'media_filename', 'text']
        )

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


def compute_wer_pointwise(ref_df, hyp_df):
    joint_df = ref_df.merge(hyp_df, on='media_filename', suffixes=('_ref', '_hyp'))
    lengths = pd.DataFrame(joint_df.apply(lambda x: {'ref_length': len(x['text_ref'].split())}, axis=1, result_type='expand'))
    joint_df = joint_df.join(lengths)
    edits = joint_df.apply(lambda x: compute_alignment(x['text_hyp'].split(), x['text_ref'].split()), axis=1, result_type='expand')
    joint_df = joint_df.join(edits)
    return joint_df


def main():
    parser = argparse.ArgumentParser(description='haw compares word errors', formatter_class=argparse.Formatter)
    parser.add_argument('-w', '--words', action='store_true', help='Compute WER in words by joining BPE tokens')
    parser.add_argument('ref', type=Path, help='ref')
    parser.add_argument('hyp', type=Path, help='hyp')
    args = parser.parse_args()

    ref_df = read_text(args.ref, join_bpe=args.words)
    hyp_df = read_text(args.hyp, join_bpe=args.words)
    joint_df = compute_wer_pointwise(ref_df, hyp_df)
    joint_df.to_csv(sys.stdout, sep='\t', index=False)
    total = joint_df['total'].sum()
    ref_length = joint_df['ref_length'].sum()
    ins = joint_df['ins'].sum()
    del_ = joint_df['del'].sum()
    sub = joint_df['sub'].sum()
    print('%WER', round(100 * total / ref_length, 2), f'errors={total}/{ref_length}', f'ins={ins}', f'del={del_}', f'sub={sub}', file=sys.stderr)


if __name__ == '__main__':
    main()
