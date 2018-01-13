import numpy as np

import tree._tree
from numpy.testing import assert_array_equal
from pytest import approx


def test_leaf_mapper():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 15.0]], dtype='float32')
    Y = np.array([1.0, 0.0, 0.0, 1.0], dtype='float32')

    lmb = tree._tree.MeanLeafMapperBuilder()

    lm = lmb.build(X, Y)

    assert_array_equal(np.array([.5, .5, .5, .5]),
                       lm.predict(X))


def test_cross_entropy():
    truth = np.array([1.0, 0.0, 0.0, 1.0], dtype='float32')
    predicted = np.array([.9, .1, .2, .8], dtype='float32')

    ce = tree._tree.CrossEntropyLoss()

    assert 0.16425204277038574 == ce.loss(truth, predicted)


def test_splitter():

    X = np.array([[1.0, 2.0],
                  [20.0, 4.0],
                  [5.0, 6.0],
                  [18.0, 15.0]], dtype='float32')
    XX = X.copy()

    Y = np.array([1.0,
                  0.0,
                  0.0,
                  1.0], dtype='float32')
    YY = Y.copy()

    lmb = tree._tree.MeanLeafMapperBuilder()
    ce = tree._tree.CrossEntropyLoss()

    (best_split, best_loss) = tree._tree.getBestSplit(XX, 0, YY, ce, lmb, set(X[0]))

    assert best_loss == approx(0.693147182465)
    assert best_split == 1.0

    # Ensure we didn't change the incoming array
    assert_array_equal(X, XX)
    assert_array_equal(Y, YY)

