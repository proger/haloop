from ha import argparse
from pathlib import Path
import pandas as pd
import sys
from kaldialign import edit_distance


def clean_tokens(text):
    return ' '.join([token for token in text.split() if token != '␣'])


def read_text(filename: Path):
    def clean(i, key, text):
        return i, key, clean_tokens(text)

    with open(filename) as f:
        return pd.DataFrame(
            [clean(i, *line.strip().split(maxsplit=1)) for i, line in enumerate(f)],
            columns=['dataset_index', 'media_filename', 'text']
        )


def compute_wer_pointwise(hyp_df, ref_df):
    joint_df = hyp_df.merge(ref_df, on='media_filename', suffixes=('_hyp', '_ref'))
    edits = joint_df.apply(lambda x: edit_distance(x['text_hyp'], x['text_ref']), axis=1, result_type='expand')
    joint_df = joint_df.join(edits)
    lengths = pd.DataFrame(joint_df.apply(lambda x: {'length': len(x['text_ref'].split())}, axis=1, result_type='expand'))
    return joint_df.join(lengths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=Path)
    parser.add_argument('hyp', type=Path)
    args = parser.parse_args()

    ref_df = read_text(args.ref)
    hyp_df = read_text(args.hyp)
    compute_wer_pointwise(hyp_df, ref_df).to_csv(sys.stdout, sep='\t', index=False)
    