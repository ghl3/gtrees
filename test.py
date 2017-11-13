
import numpy as np

import tree._my_tree
from numpy.testing import assert_array_equal


def test_leaf_mapper():

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 15.0]], dtype='float32')
    Y = np.array([1.0, 0.0, 0.0, 1.0], dtype='float32')

    lm = tree._my_tree.MeanLeafMapper()

    lm.init(X, Y)

    assert_array_equal(np.array([.5, .5, .5, .5]),
                       lm.predict(X))
