import numpy as np

import tree._my_tree
from numpy.testing import assert_array_equal


def test_leaf_mapper():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 15.0]], dtype='float32')
    Y = np.array([1.0, 0.0, 0.0, 1.0], dtype='float32')

    lmb = tree._my_tree.MeanLeafMapperBuilder()

    lm = lmb.build(X, Y)

    assert_array_equal(np.array([.5, .5, .5, .5]),
                       lm.predict(X))


def test_cross_entropy():
    truth = np.array([1.0, 0.0, 0.0, 1.0], dtype='float32')
    predicted = np.array([.9, .1, .2, .8], dtype='float32')

    ce = tree._my_tree.CrossEntropyLoss()

    assert 0.2369655966758728 == ce.loss(truth, predicted)


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

    lmb = tree._my_tree.MeanLeafMapperBuilder()
    ce = tree._my_tree.CrossEntropyLoss()
    spliter = tree._my_tree.SpitFinder()

    (best_split, best_loss) = spliter.getBestSplit(0, XX, YY, lmb, ce)

    print "Best Split: {} Best Loss: {}".format(best_split, best_loss)

    # Ensure we didn't change the incoming array
    assert_array_equal(X, XX)
    assert_array_equal(Y, YY)


##X
 #   assert 0.2369655966758728 == ce.loss(truth, predicted)
