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
parser.add_argument('--log', type=Path, default=Path(',18a09a848c2.3343822'),
                    help='log of the training run')

def read_text(filename: Path):
    with open(filename) as f:
        return pd.DataFrame([line.strip().split(maxsplit=1) for line in f], columns=['media_filename', 'text'])

def read_grads(filename: Path):
    return pd.read_csv(filename, sep='\t', header=None, names=['stub', 'dataset_index', 'grad_norm', 'loss'])

def training_log_to_dataset(training_log_filename: Path):
    "reads output of hac using heuristics to extract the dataset"
    train_hypotheses = []
    with open(training_log_filename) as f:
        decoding_train = False
        for line in f:
            if decoding_train and line.startswith('12') and 'hyp' in line:
                epoch, dataset_index, hypN, text = line.strip().split('\t')
                assert epoch == "12" and hypN.startswith('hyp'), f"epoch={epoch}, hypN={hypN}"
                train_hypotheses.append((int(dataset_index), text))
            elif line.startswith('valid [12'):
                decoding_train = True
                continue
    df = pd.DataFrame(train_hypotheses, columns=['dataset_index', 'hyp_text'])
    df.sort_values(by='dataset_index', ascending=True, inplace=True)
    return df.set_index('dataset_index')


if __name__ == '__main__':
    args = parser.parse_args()

    oracle = read_text(args.oracle)
    corrupted = read_text(args.corrupted)

    train_hypotheses = training_log_to_dataset(args.log)
    grad_norms_dataset = train_hypotheses.join(corrupted)
    grad_norms_dataset[['media_filename', 'hyp_text']].to_csv(args.exp / 'hyp.txt.piece', sep='\t', header=False, index=False)

    if not (args.exp / 'grads.txt').exists():
        print('please compute gradient norms by running:', file=sys.stderr)
        print(
            'nq bash -c "hac',
            f'--grad-norms fbank:{args.exp / "hyp.txt.piece"}',
            '--device cuda:1',
            '--init', args.exp / 'last.pt',
            '--vocab exp/libribpe.vocab --compile | tee', args.exp / 'grads.txt',
            '"'
        )
        sys.exit(1)

    grad_norms_result = read_grads(args.exp / 'grads.txt')

    # Compute log-space EGL for each utterance
    grad_norms = pd.concat([
        grad_norms_dataset.reset_index(),
        grad_norms_result
    ], axis=1)

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
    # TODO: concat clean.txt.piece from previous experiments
    print(
        'nq hac --train',
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
    print(
        'python -m ha.query_sim',
        '--oracle', args.oracle,
        '--corrupted', args.exp / 'corrupted.txt.piece',
        '--exp', next_exp,
        '--log', '???',
    )