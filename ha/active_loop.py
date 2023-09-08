from ha import argparse
from pathlib import Path
import pandas as pd
import sys
from ha.subprocess import run
import numpy as np

from .wer import compute_wer_pointwise, clean_tokens, read_text, format_wer

parser = argparse.ArgumentParser(description="""Learning to improve supervision.
""", formatter_class=argparse.Formatter)
parser.add_argument('--oracle', type=Path, default=Path('data/flaky/train-clean-100.ref.txt.piece'),
                    help='dataset with true labels')
# parser.add_argument('--oracle-dirty', type=Path, default=Path('exp/active/mini-egl/onlydirty.txt.piece'),
#                     help='dataset with true dirty labels')
parser.add_argument('--query-size', type=str, default='10h', # '2196'
                    help='number of utterances or hours (with h at the end, like 10h) to query')
parser.add_argument('--initial-corrupted', type=Path, default=Path('data/flaky/train-clean-100.dirty28538.txt.piece'),
                    help='initial dataset with corrupted labels')
parser.add_argument('--eval', type=Path, default=Path('data/flaky/dev-clean.txt.piece'),
                    help='evaluation dataset')
parser.add_argument('--vocab', type=Path, default=Path('data/flaky/libribpe.vocab'),
                    help='vocab file')
parser.add_argument('--duration', type=Path, default=Path('data/flaky/train-clean-100.seconds'),
                    help='duration file (filename TAB seconds)')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

#parser.add_argument('--strategy', type=str, choices=['random', 'egl', 'oracle-max-wer', 'long', 'entropy', 'prob', 'spin'], default='random', help='query strategy')
parser.add_argument('strategy', nargs='+', help='query strategy')

# iteration parameters
parser.add_argument('--start', type=int, default=0,
                    help='start iteration')
parser.add_argument('--steps', type=int, default=10,
                    help='iterations to make since --start')
parser.add_argument('--exp', type=Path, default=Path('exp/random'),
                    help='experiment root for all iterations')
parser.add_argument('--train', action='store_true',
                    help='train the model after every query')


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
        decoding_epoch = None
        for line in f:
            if decoding_epoch and line.startswith(decoding_epoch) and 'hyp' in line:
                epoch, dataset_index, hypN, text = line.strip().split('\t')
                assert epoch == decoding_epoch and hypN.startswith('hyp'), f"epoch={epoch}, hypN={hypN}"
                train_hypotheses.append((int(dataset_index), clean_tokens(text)))
            elif line.startswith('testing'):
                decoding_epoch = line.strip().split(maxsplit=1)[1]
                continue
            elif line.startswith('valid [12'):
                decoding_epoch = '12'
                continue
    df = pd.DataFrame(train_hypotheses, columns=['dataset_index', 'hyp_text'])
    df.sort_values(by='dataset_index', ascending=True, inplace=True)
    return df.set_index('dataset_index')


def test_log_to_dataset(test_log_filename: Path):
    "reads output of hac using heuristics to extract the dataset with log probs and entropies"
    hypotheses = []
    with open(test_log_filename) as f:
        decoding_epoch = None
        for line in f:
            if line.startswith('testing'):
                decoding_epoch = line.strip().split(maxsplit=1)[1]
            elif decoding_epoch and line.startswith(decoding_epoch) and '\thyp' in line:
                epoch, dataset_index, hypN, last_label = line.strip().split('\t')
                assert epoch == decoding_epoch, f"epoch={epoch}"
            elif decoding_epoch and line.startswith(decoding_epoch) and '\tstat' in line:
                epoch, dataset_index, statN, text_stat = line.strip().split('\t')
                assert epoch == decoding_epoch, f"epoch={epoch}"
                stat = dict([kv.split('=') for kv in text_stat.split(' ')])
                hypotheses.append((int(dataset_index), float(stat.get('log_prob')), float(stat.get('log_prob_per_token')), float(stat.get('entropy_per_token')), stat.get('prompt', '<s>'), last_label))
    df = pd.DataFrame(hypotheses, columns=['dataset_index', 'log_prob', 'log_prob_per_token', 'entropy_per_token', 'prompt', 'text'])
    df.sort_values(by='dataset_index', ascending=True, inplace=True)
    return df.set_index('dataset_index')


def estimate_egl(
    grad_norms_df: pd.DataFrame # ['grad_norm', 'loss', 'media_filename']
) -> pd.Series:
    #from scipy.special import logsumexp
    # #
    # #    \log \sum_y ||\grad P(y|x)||**2 P(y|x)
    # # =  \log \sum_y exp(\log ||\grad P(y|x)||**2 - NLL(y|x))
    # #
    # grad_norms['product'] = np.log((grad_norms['grad_norm'] ** 2)) - grad_norms['loss']
    # egl = grad_norms.groupby('media_filename')['product'].apply(logsumexp)

    grad_norms_df['product'] = (grad_norms_df['grad_norm'] ** 2) * np.exp(-grad_norms_df['loss'])
    egl = grad_norms_df.groupby('media_filename')['product'].apply(np.sum)

    egl.sort_values(ascending=False, inplace=True)
    return egl



def train(root, train, eval, test, args, spin=False, test_attempts=1):
    root.mkdir(exist_ok=True, parents=True)
    if not (root / 'last.pt').exists() or not (root / 'train.log').exists():
        prefixes = ['mask:fbank:speed:', 'mask:fbank:speed:randpairs:']
        run([
            'hac',
            '--train', ','.join([f'{prefix}{train}' for prefix in prefixes]),
            '--eval', f'fbank:{eval}', ] + ([
            '--test', f'fbank:{test}',
            '--test-attempts', str(test_attempts),
            ] if test else []) + f'--num-epochs 13 --num-workers 16 --lr_decay_iters 15835 --lr_schedule linear --warmup_iters 3000 --batch-size 48 --lr 0.0006 --min_lr 0 --eval-batch-size 1024 --compile --vocab {str(args.vocab)} --weight_decay 0.1'.split() + [
            '--exp', str(root),# '--allow-oom'
            ] + (["--test-spin-prompts", "--arch", "transformer:514"] if spin else []) + [
            '--device', args.device,
        ], output_filename=root / 'train.log')
        just_trained = True
    else:
        just_trained = False
    return just_trained


def perform_query(ranked_df, query_size: str):
    if query_size.endswith('h'):
        return query_hours(ranked_df, max_seconds=int(query_size[:-1]) * 60 * 60)
    else:
        return ranked_df.head(int(query_size))


def query_hours(ranked_df, max_seconds=10*60*60):
    end = 0
    seconds = 0.
    while end < len(ranked_df):
        end += 1
        seconds += ranked_df.iloc[end].seconds
        if seconds > max_seconds:
            break
    return ranked_df.iloc[:end].set_index('media_filename')


def perform_egl(args, exp, combined_train, corrupted, prev_corrupted_dataset):
    """
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
    """
    just_trained = train(exp, combined_train, args.eval, corrupted, args, test_attempts=20)

    train_hypotheses = training_log_to_dataset(exp / 'train.log')
    grad_norms_dataset = train_hypotheses.join(prev_corrupted_dataset)

    if not (exp / 'grads.txt').exists() or just_trained:
        print('# writing', exp / 'hyp.txt.piece', file=sys.stderr)
        grad_norms_dataset[['media_filename', 'hyp_text']].to_csv(exp / 'hyp.txt.piece', sep='\t', header=False, index=False)
        print('# computing gradient norms', file=sys.stderr)
        run([
            'hac',
            '--grad-norms', f'fbank:{exp / "hyp.txt.piece"}',
            '--device', args.device,
            '--init', str(exp / 'last.pt'),
            '--vocab', str(args.vocab),
            '--compile',
        ], output_filename=exp / 'grads.txt')
    else:
        print('# using existing', exp / 'grads.txt', file=sys.stderr)
        run(["wc", "-l", str(exp / 'grads.txt')])

    grad_norms_result = read_grads(exp / 'grads.txt')

    # Compute EGL for each utterance
    grad_norms_df = pd.concat([
        grad_norms_dataset.reset_index(),
        grad_norms_result
    ], axis=1)

    query = estimate_egl(grad_norms_df)
    query.to_csv(exp / 'egl', sep='\t', header=False)
    print('# writing utterance scores to', exp / 'egl', file=sys.stderr)
    return query


def run_step(args, exp, *, prev=None, is_final=False):
    oracle = read_text(args.oracle)
    duration = pd.read_csv(args.duration, sep='\t', names=['media_filename', 'seconds'])

    if prev is not None:
        print('# continuing from', prev, 'in', exp, file=sys.stderr)

        combined_train = prev / 'combined_train.txt.piece'
        assert combined_train.exists(), f'{combined_train} does not exist'
        corrupted = prev / 'corrupted.txt.piece'
        prev_corrupted_dataset = read_text(corrupted)
    else:
        print('# starting from scratch', exp, file=sys.stderr)
        corrupted = combined_train = args.initial_corrupted
        prev_corrupted_dataset = read_text(args.initial_corrupted)

    exp.mkdir(exist_ok=True, parents=True)

    match args.strategy:
        case ['random']:
            query = prev_corrupted_dataset.sample(frac=1, replace=False, random_state=args.seed)
        case ['oracle-max-wer']:
            oracle_wer_df = compute_wer_pointwise(prev_corrupted_dataset, oracle)
            oracle_wer_df['text'] = oracle_wer_df['text_ref']
            query = oracle_wer_df.sort_values('total', ascending=False)
        case ['long']:
            query = prev_corrupted_dataset.copy()
            query['sizes'] = query['text'].str.count(' ') + 1
            query = query.sort_values(by='sizes', ascending=False)
        case ['entropy']:
            train(exp / 'entropy_prob', combined_train, args.eval, args.oracle, args) # why oracle?

            entropy_prob_df = test_log_to_dataset(exp / 'entropy_prob/train.log')
            entropy_prob_df = pd.concat([
                oracle,
                entropy_prob_df
            ], axis=1)
            query = entropy_prob_df.sort_values('entropy_per_token', key=lambda x: x.astype(float), ascending=False)
        case ['prob']:
            train(exp / 'entropy_prob', combined_train, args.eval, args.oracle, args) # why oracle?

            entropy_prob_df = pd.concat([
                oracle,
                test_log_to_dataset(exp / 'entropy_prob/train.log')
            ], axis=1)
            query = entropy_prob_df.sort_values('log_prob_per_token', key=lambda x: -x.astype(float), ascending=False)
        case ['spin']:
            test = combined_train # TODO: test only on unknown items in the training set
            train(exp / 'spin', combined_train, args.eval, combined_train, args, spin=True)

            # order by log_prob_per_token under <↓> condition

            spin_df = test_log_to_dataset(exp / 'spin/train.log')
            spin_df = spin_df[spin_df['prompt'] == '<↓>']
            spin_df = read_text(test).merge(spin_df, on='dataset_index')
            query = spin_df.sort_values('log_prob_per_token', key=lambda x: -x.astype(float), ascending=False)
        case ['egl']:
            query = perform_egl(args, combined_train, corrupted, prev_corrupted_dataset)
        case ['logfile', log_filename, test_dataset]:
            # read utterance log probs from file
            df = test_log_to_dataset(Path(log_filename))
            df1 = df.groupby(df.index).log_prob.mean().rename('log_prob_mean')
            df = read_text(Path(test_dataset)).merge(df1, on='dataset_index')
            query = prev_corrupted_dataset.set_index('media_filename').merge(df.set_index('media_filename'), left_index=True, right_index=True)
            query['text'] = query['text_x']
            del query['text_x']
            del query['text_y']
            query = query.reset_index()
            query = query.sort_values('log_prob_mean', key=lambda x: -x.astype(float), ascending=False)

    print(query, flush=True)
    query = query[['media_filename', 'text']]
    query = query.set_index('media_filename')
    query = query.merge(duration, on='media_filename')
    if is_final:
        # take all remaining data
        print('# final query. queried', len(query), 'clean utterances, query size was', args.query_size, file=sys.stderr)
    else:
        query = perform_query(query, query_size=args.query_size)
        print('# queried', len(query), 'clean utterances, query size was', args.query_size, file=sys.stderr)
    assert len(query) > 0, "query size is zero, something is wrong"
    assert len(query) < 10000, "query size is too large, something is wrong"

    # Read true labels for the query from the oracle dataset
    oracle_query_result = oracle[oracle['media_filename'].isin(query.index)]
    print('# writing', exp / 'query_result.txt.piece', file=sys.stderr)
    oracle_query_result.to_csv(exp / 'query_result.txt.piece', sep='\t', header=False, index=False)

    if prev:
        # Concat clean.txt.piece from previous experiments
        clean_train_dataset = pd.concat([read_text(prev / 'clean.txt.piece'), oracle_query_result])
    else:
        clean_train_dataset = oracle_query_result

    clean_train_dataset.to_csv(exp / 'clean.txt.piece', sep='\t', header=False, index=False)
    print('# writing', exp / 'clean.txt.piece', file=sys.stderr)

    print('# computing errors between oracle query result and previously corrupted dataset', file=sys.stderr)
    ler_df = compute_wer_pointwise(
        oracle_query_result[['media_filename', 'text']],
        prev_corrupted_dataset[['media_filename', 'text']]
    )
    print(*format_wer(ler_df, tag='LER'), file=sys.stderr)
    wer_df = compute_wer_pointwise(
        oracle_query_result[['media_filename', 'text']],
        prev_corrupted_dataset[['media_filename', 'text']],
        join_bpe=True
    )
    print(*format_wer(wer_df), file=sys.stderr)

    # Read the rest of the labels from the original dataset
    remaining_corrupted_dataset = prev_corrupted_dataset[~prev_corrupted_dataset['media_filename'].isin(query.index)]
    remaining_corrupted_dataset.to_csv(exp / 'corrupted.txt.piece', sep='\t', header=False, index=False)

    combined_train_new_path = exp / 'combined_train.txt.piece'
    print('# writing combined dataset', combined_train_new_path, file=sys.stderr)
    combined_train = pd.concat([clean_train_dataset, remaining_corrupted_dataset])
    combined_train.to_csv(combined_train_new_path, sep='\t', header=False, index=False)

    print('# computing errors between new combined dataset and oracle', file=sys.stderr)
    gler_df = compute_wer_pointwise(
        combined_train[['media_filename', 'text']],
        oracle
    )
    print(*format_wer(gler_df, tag='GLER'), file=sys.stderr)

    gwer_df = compute_wer_pointwise(
        combined_train[['media_filename', 'text']],
        oracle,
        join_bpe=True
    )
    print(*format_wer(gwer_df, tag='GWER'), file=sys.stderr)

    return combined_train_new_path


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)

    for step in range(args.start, args.start+args.steps):
        exp = args.exp / f'{step:02d}'
        if step == 0:
            train_path = run_step(args, exp)
        else:
            prev = args.exp / f'{step-1:02d}'
            train_path = run_step(args, exp, prev=prev, is_final=step == args.start+args.steps-1)

        if args.train:
            train(exp / 'post', train=train_path, eval=args.eval, test=None, args=args)

if __name__ == '__main__':
    main()
