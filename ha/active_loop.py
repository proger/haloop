import argparse
from pathlib import Path
import pandas as pd
import sys
import io
from ha.subprocess import run

parser = argparse.ArgumentParser(description="""Active learning on noisy labels.

Given a training set, train a model, and decode the training set to
get multiple labels per utterance.

Given a dataset with multiple labels per utterance, compute the gradient
norm and loss value for each label-utterance pair.

From a list of gradient norms and losses for each label-utterance pair,
compute the expected gradient length (EGL) for each utterance:

    EGL(x) = \sum_y P(y|x) ||\grad P(y|x)||**2

Based on largest EGL values, select a batch utterances to query.

Then, fulfill the query by reading true labels from the oracle dataset
and the rest of the labels from the original dataset.
""")
parser.add_argument('--oracle', type=Path, default=Path('data/corrupted-librispeech/train-clean-100.ref.txt.piece'),
                    help='dataset with true labels')
parser.add_argument('--query-size', type=int, default=2196,
                    help='number of utterances to query')
parser.add_argument('--initial-corrupted', type=Path, default=Path('data/corrupted-librispeech/train-clean-100.dirty28538.txt.piece'),
                    help='initial dataset with corrupted labels')
parser.add_argument('--prev', type=Path, required=False,
                    help='experiment directory')
parser.add_argument('--exp', type=Path, default=Path('exp/active/egl/01'),
                    help='experiment directory')

def clean_tokens(text):
    return ' '.join([token for token in text.split() if token != '␣'])

def read_text(filename: Path):
    def clean(key, text):
        return key, clean_tokens(text)

    with open(filename) as f:
        return pd.DataFrame(
            [clean(*line.strip().split(maxsplit=1)) for line in f],
            columns=['media_filename', 'text']
        )

def read_grads(filename: Path):
    rows = []
    with open(filename) as f:
        for line in f:
            if not line.startswith('grad_norm,loss'):
                continue
            stub, dataset_index, grad_norm, loss = line.strip().split('\t')
            rows.append((int(dataset_index), float(grad_norm), float(loss)))
    return pd.DataFrame(rows, columns=['dataset_index', 'grad_norm', 'loss']).set_index('dataset_index')

def training_log_to_dataset(training_log_filename: Path):
    "reads output of hac using heuristics to extract the dataset"
    train_hypotheses = []
    with open(training_log_filename) as f:
        decoding_train = False
        for line in f:
            if decoding_train and line.startswith('12') and 'hyp' in line:
                epoch, dataset_index, hypN, text = line.strip().split('\t')
                assert epoch == "12" and hypN.startswith('hyp'), f"epoch={epoch}, hypN={hypN}"
                train_hypotheses.append((int(dataset_index), clean_tokens(text)))
            elif line.startswith('valid [12'):
                decoding_train = True
                continue
    df = pd.DataFrame(train_hypotheses, columns=['dataset_index', 'hyp_text'])
    df.sort_values(by='dataset_index', ascending=True, inplace=True)
    return df.set_index('dataset_index')


if __name__ == '__main__':
    args = parser.parse_args()

    oracle = read_text(args.oracle)

    if args.prev:
        combined_train = args.prev / 'combined_train.txt.piece'
        assert combined_train.exists(), f'{combined_train} does not exist'
        corrupted = read_text(args.prev / 'corrupted.txt.piece')
    else:
        print('# starting from scratch', file=sys.stderr)
        combined_train = args.initial_corrupted
        corrupted = read_text(args.initial_corrupted)

    args.exp.mkdir(exist_ok=True, parents=True)

    if not (args.exp / 'last.pt').exists() or not (args.exp / 'train.log').exists():
        prefixes = ['mask:fbank:speed:', 'mask:fbank:speed:randpairs:']
        run([
            'hac',
            '--train', ','.join([prefix + str(combined_train) for prefix in prefixes]),
            '--eval', 'fbank:data/corrupted-librispeech/dev-clean.txt.piece',
            '--test-attempts', '20',
            '--test', f'fbank:{combined_train}'
            ] + '--num-epochs 13 --num-workers 16 --lr_decay_iters 15835 --lr_schedule linear --warmup_iters 3000 --device cuda:1 --batch-size 48 --lr 0.0006 --min_lr 0 --eval-batch-size 1024 --compile --vocab exp/libribpe.vocab --weight_decay 0.1'.split() + [
            '--exp', f'{args.exp}', '--allow-oom',
        ], output_filename=args.exp / 'train.log')
        just_trained = True
    else:
        just_trained = False

    train_hypotheses = training_log_to_dataset(args.exp / 'train.log')
    grad_norms_dataset = train_hypotheses.join(corrupted)
    grad_norms_dataset[['media_filename', 'hyp_text']].to_csv(args.exp / 'hyp.txt.piece', sep='\t', header=False, index=False)

    if not (args.exp / 'grads.txt').exists() or just_trained:
        print('# computing gradient norms', file=sys.stderr)
        run([
            'hac',
            '--grad-norms', f'fbank:{args.exp / "hyp.txt.piece"}',
            '--device', 'cuda:1',
            '--init', str(args.exp / 'last.pt'),
            '--vocab', 'exp/libribpe.vocab', '--compile',
        ], output_filename=args.exp / 'grads.txt')

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
    print('# writing utterance scores to', args.exp / 'egl', file=sys.stderr)

    query = egl[:args.query_size]

    # Read true labels for the query from the oracle dataset
    oracle_query = oracle[oracle['media_filename'].isin(query.index)]
    if args.prev:
        # Concat clean.txt.piece from previous experiments
        oracle_query = pd.concat([read_text(args.prev / 'clean.txt.piece'), oracle_query])

    print('# querying', len(query), 'clean utterances')
    oracle_query.to_csv(args.exp / 'clean.txt.piece', sep='\t', header=False, index=False)
    print('# writing', args.exp / 'clean.txt.piece', file=sys.stderr)

    # Read the rest of the labels from the original dataset
    corrupted_rest = corrupted[~corrupted['media_filename'].isin(query.index)]
    corrupted_rest.to_csv(args.exp / 'corrupted.txt.piece', sep='\t', header=False, index=False)

    print('# writing combined dataset', file=sys.stderr)
    combined_train = pd.concat([oracle_query, corrupted_rest])
    combined_train.to_csv(args.exp / 'combined_train.txt.piece', sep='\t', header=False, index=False)

    try:
        next_exp = args.exp.parent / f'{int(args.exp.name) + 1:02}'
        print(
            'python -m ha.active_loop',
            '--prev', args.exp,
            '--exp', next_exp,
        )
    except ValueError:
        pass