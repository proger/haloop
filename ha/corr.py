import numpy as np
import sys

from . import argparse

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