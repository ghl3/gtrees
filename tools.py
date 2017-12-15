
import math
import random
import numpy as np
import pandas as pd
from sklearn import datasets


def make_hastie_sample(n_samples):

    features, targets = datasets.make_hastie_10_2(n_samples=n_samples)

    features = pd.DataFrame(features, columns=['feature_{}'.format(i) for i in range(features.shape[1])], dtype=np.float32)
    targets = pd.Series(targets, name='target', dtype=np.float32)
    targets = pd.Series(targets.map(lambda x: 1.0 if x > 0 else 0.0), dtype=np.float32)
    return features, targets


def make_kddcup(n_samples):

    features, targets = datasets.fetch_kddcup99(subset='smtp')

    features = pd.DataFrame(features, columns=['feature_{}'.format(i) for i in range(features.shape[1])], dtype=np.float32)
    targets = pd.Series(targets, name='target', dtype=np.float32)
    targets = targets.map(lambda x: 1.0 if x > 0 else 0.0)

    features = featurse.sample(n=n_samples)

    return features, targets.loc[features.index]


def make_random_classification(n_samples, n_features=100):



    features, targets = datasets.make_classification(n_samples=n_samples,
                                                     n_features=n_features,
                                                     n_informative=8,
                                                     n_classes=2,
                                                     n_clusters_per_class=4)


    features = pd.DataFrame(features, columns=['feature_{}'.format(i) for i in range(features.shape[1])], dtype=np.float32)
    targets = pd.Series(targets, name='target', dtype=np.float32)
    targets = targets.map(lambda x: 1.0 if x > 0 else 0.0)

    return features, targets.loc[features.index]
