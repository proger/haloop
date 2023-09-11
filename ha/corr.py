import numpy as np
import sys

from . import argparse

def rank_corr(l, r):
    "spearman rank correlation between two differently ordered dataframes with the same index"
    l['left_rank'] = np.arange(len(l))
    r['right_rank'] = np.arange(len(r))
    both = l.merge(r, left_index=True, right_index=True)
    rank_sq_diff = (both['left_rank'] - both['right_rank'])**2
    return 1 - 6 * rank_sq_diff.sum() / (len(both) * (len(both)**2 - 1))

def main():
    parser = argparse.ArgumentParser(description='hax computes dependence statistics', formatter_class=argparse.Formatter)
    args = parser.parse_args()

    data = np.loadtxt(sys.stdin, delimiter=' ')

    column1 = data[:, 0]
    column2 = data[:, 1]

    correlation_coefficient = np.corrcoef(column1, column2)[0, 1]

    print(correlation_coefficient)


if __name__ == '__main__':
    main()