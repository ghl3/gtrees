from __future__ import division

import argparse

import numpy as np
import pandas as pd

from sklearn import datasets

import gtree
import tools


def make_hastie_sample(n_samples):
    features, targets = datasets.make_hastie_10_2(n_samples=n_samples)

    features = pd.DataFrame(features, columns=['feature_{}'.format(i) for i in range(features.shape[1])])
    targets = pd.Series(targets, name='target')
    targets = targets.map(lambda x: 1.0 if x > 0 else 0.0)
    return features, targets


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples')

    parser.add_argument('--max-depth', type=int, default=1000,
                        help='Max depth of trees to grow')

    parser.add_argument('--num-iterations', type=int, default=1,
                        help='Number of iterations')

    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='Loss Function to use')

    parser.add_argument('--leaf-prediction', type=str, default='mean',
                        help='Leaf Prediction Function')

    args = parser.parse_args()

    for _ in range(args.num_iterations):
        features, targets = tools.make_random_classification(args.num_samples)
        features = pd.DataFrame(features, dtype=np.float32)
        targets = pd.Series(targets, dtype=np.float32)

        # features, targets = make_hastie_sample(args.num_samples)
        gtree.train_greedy_tree(features,
                                targets,
                                loss=args.loss,
                                max_depth=args.max_depth,
                                leaf_prediction=args.leaf_prediction)


if __name__ == '__main__':
    main()
