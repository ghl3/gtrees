from __future__ import division

import argparse
import pandas as pd

from sklearn import datasets

import gtree


def make_hastie_sample(n_samples):
    features, targets = datasets.make_hastie_10_2(n_samples=n_samples)

    features = pd.DataFrame(features, columns=['feature_{}'.format(i) for i in range(features.shape[1])])
    targets = pd.Series(targets, name='target')
    targets = targets.map(lambda x: 1.0 if x > 0 else 0.0)
    return features, targets


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples')

    parser.add_argument('--max-depth', type=int, default=1000,
                        help='Max depth of trees to grow')

    parser.add_argument('--num-iterations', type=int, default=1,
                        help='Number of iterations')

    args = parser.parse_args()

    for _ in range(args.num_iterations):
        features, targets = make_hastie_sample(args.num_samples)
        gtree.train_greedy_tree(features, targets, loss_fn=gtree.error_rate_loss, max_depth=args.max_depth)


if __name__ == '__main__':
    main()
