import argparse
from pathlib import Path
import pandas as pd
import sys

parser = argparse.ArgumentParser(description="""Simulate dataset queries.

Given a dataset with multiple labels per utterance,
a list of gradient norms and losses for each label-utterance pair,
compute the expected gradient length (EGL) for each utterance.

EGL(x) = \sum_y P(y|x) ||\grad P(y|x)||**2

Based on highest EGL values, select a batch utterances to query.

Then, fulfill the query by reading true labels from the oracle dataset
and the rest of the labels from the original dataset.
""")
parser.add_argument('--oracle', type=Path, default=Path('data/corrupted-librispeech/train-clean-100.ref.txt.piece'),
                    help='dataset with true labels')
parser.add_argument('--corrupted', type=Path, default=Path('data/corrupted-librispeech/train-clean-100.dirty28538.txt.piece'),
                    help='initial dataset with dirty labels')
parser.add_argument('--query-size', type=int, default=2196,
                    help='number of utterances to query')
parser.add_argument('--exp', type=Path, default=Path('exp/active/egl/01'),
                    help='experiment directory')
parser.add_argument('--grad-norms-dataset', type=Path, default=Path('exp/active/egl/01/hyp.txt.piece'),
                    help="""Dataset with multiple candidate labels per utterance.

Computed like this:
# ,18a053800d0.1720196 is output of https://wandb.ai/stud76/ha/runs/wgr0e8xn
awk '/valid \[12/{x=1} x' ,18a053800d0.1720196 | grep '^12' | grep hyp | sort -n -k2,3 | gzip -c > exp/active/egl/01/hyp.txt.gz
paste <(seq 0 28537) data/corrupted-librispeech/train-clean-100.dirty28538.txt.piece | awk '{print $1, $2}' > exp/active/egl/01/id2flac
zcat exp/active/egl/01/hyp.txt.gz | cut -f2,4 | join exp/active/egl/01/id2flac - | cut -d' ' -f2- > exp/active/egl/01/hyp.txt.piece
""")
parser.add_argument('--grad-norms-result', type=Path, default=Path('exp/active/egl/01/grads.txt'),
                    help="""File with output of hac --grad-norms:

hac --grad-norms fbank:exp/active/egl/01/hyp.txt.piece --device cuda:0 --init exp/active/egl/01/last.pt --vocab exp/libribpe.vocab --compile | tee exp/active/egl/01/grads.txt
""")

def read_text(filename: Path):
    with open(filename) as f:
        return pd.DataFrame([line.strip().split(maxsplit=1) for line in f], columns=['media_filename', 'text'])

def read_grads(filename: Path):
    return pd.read_csv(filename, sep='\t', header=None, names=['stub', 'dataset_index', 'grad_norm', 'loss'])


if __name__ == '__main__':
    args = parser.parse_args()

    grad_norms_dataset = read_text(args.grad_norms_dataset)
    grad_norms_result = read_grads(args.grad_norms_result)
    oracle = read_text(args.oracle)
    corrupted = read_text(args.corrupted)

    # Compute log-space EGL for each utterance
    grad_norms = pd.concat([grad_norms_dataset, grad_norms_result], axis=1)

    import numpy as np
    from scipy.special import logsumexp
    #
    #    \log \sum_y ||\grad P(y|x)||**2 P(y|x) 
    # =  \log \sum_y exp(\log ||\grad P(y|x)||**2 - NLL(y|x))
    #
    grad_norms['product'] = np.log((grad_norms['grad_norm'] ** 2)) - grad_norms['loss']

    egl = grad_norms.groupby('media_filename')['product'].apply(logsumexp)
    egl.sort_values(ascending=False, inplace=True)

    egl.to_csv(args.exp / 'egl', sep='\t', header=False)
    print('writing utterance scores to', args.exp / 'egl', file=sys.stderr)

    query = egl[:args.query_size]

    # Read true labels for the query from the oracle dataset
    oracle_query = oracle[oracle['media_filename'].isin(query.index)]

    print('querying', len(query), 'clean utterances')
    oracle_query.to_csv(args.exp / 'clean.txt.piece', sep='\t', header=False, index=False)
    print('writing ', args.exp / 'clean.txt.piece', file=sys.stderr)

    # Read the rest of the labels from the original dataset
    corrupted_rest = corrupted[~corrupted['media_filename'].isin(query.index)]
    corrupted_rest.to_csv(args.exp / 'corrupted.txt.piece', sep='\t', header=False, index=False)

    next_exp = args.exp.parent / f'{int(args.exp.name) + 1:02}'
    prefixes = ['mask:fbank:speed:', 'mask:fbank:speed:randpairs:']
    print(
        'hac --train',
        ','.join([prefix + file for prefix in prefixes for file in [
        str(args.exp / 'clean.txt.piece'),
        str(args.exp / 'corrupted.txt.piece'),
        ]]),
        '--eval fbank:data/corrupted-librispeech/dev-clean.txt.piece',
        '--test-attempts 20',
        f'--test fbank:{args.exp}/corrupted.txt.piece',
        '--num-epochs 13 --num-workers 16 --lr_decay_iters 15835 --lr_schedule linear --warmup_iters 3000 --device cuda:1 --batch-size 48 --lr 0.0006 --min_lr 0 --eval-batch-size 1024 --compile --vocab exp/libribpe.vocab --weight_decay 0.1',
        f'--exp {next_exp}',
    )