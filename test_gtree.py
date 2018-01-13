import pandas as pd
from numpy.testing import assert_array_equal
from pytest import approx

import gtree

import numpy as np

import tree._my_tree


class StaticLeaf(object):
    def __init__(self, val):
        self.val = val

    def predict(self, df):
        return np.array([self.val for _ in range(len(df))])


def test_tree_walking():
    # Create test data
    data = pd.DataFrame({'A': [0.1, 10, .02],
                         'B': [10, 20, 30]},
                        index=['foo', 'bar', 'baz'])

    # Create a tree that splits on variable 'A'
    t = gtree.BranchNode('A', 0.5, None, None)
    t.left = gtree.LeafNode()  # 'A', 0.5, 10, 20)
    t.right = gtree.LeafNode()  # 'A', 0.5, 100, 0)

    # Create a dummy leaf map
    leaf_map = {hash(t.left): StaticLeaf('LEFT'),
                hash(t.right): StaticLeaf('RIGHT')}

    assert {'baz': 'LEFT', 'foo': 'LEFT', 'bar': 'RIGHT'} == dict(t.predict(data, leaf_map))


def test_error_rate_loss():
    threshold = 0.5
    truth = pd.Series([1, 0, 1], dtype=np.float32)
    predicted = pd.Series([0, 1, 1], dtype=np.float32)
    loss = tree._my_tree.ErrorRateLoss(threshold)

    encountered_loss = gtree.loss(truth, predicted, type=loss)

    expected_loss = 1.0 - ((predicted >= threshold) == truth).mean()

    assert expected_loss == approx(encountered_loss)


def test_cross_entropy_loss():
    truth = pd.Series([1, 0, 1, 0, 1], dtype=np.float32)
    predicted = pd.Series([0.1, .9, 0.2, .3, .88], dtype=np.float32)
    loss = tree._my_tree.CrossEntropyLoss()

    encountered_loss = gtree.loss(truth, predicted, type=loss)

    from sklearn.metrics import log_loss

    expected_loss = log_loss(truth, predicted)  # 1.0 - ((predicted >= threshold) == truth).mean()

    assert expected_loss == approx(encountered_loss, rel=0.001)


def test_mean_leaf_builder():
    # Create test features and target
    df = pd.DataFrame({'A': [1, 2, 3],
                       'B': [10, 20, 50]}, dtype=np.float32)
    target = pd.Series([0, 0, 1], dtype=np.float32)

    # The leaf prediction is the mean good rate in his set of leaves, or 1/3
    builder = gtree.get_leaf_predictor(df.values, target.values, type='mean')
    assert np.array([0.33333, 0.33333, 0.33333]) == approx(builder.predict(df.values), rel=0.001)


def test_logit_leaf_builder():
    # Create test features and target
    df = pd.DataFrame({'A': [ 1,  1,  1,  1, 1,  0,  0,  0,  0, 0],
                       'B': [.2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2]}, dtype=np.float32)
    target = pd.Series([0, 0, 1, 0, 0, 1, 1, 1, 0, 1], dtype=np.float32)

    # The leaf prediction is a logit of the features within each leaf node
    predictor = gtree.get_leaf_predictor(df.values, target.values, type='logit')
    coefs = predictor.get_coeficients()

    # Assert that the coeficient directions make sense
    assert coefs[0] < 0
    assert coefs[1] > 0



def test_tree_equality():
    X = gtree.BranchNode('A', 0.5, None, None)
    X.left = gtree.LeafNode()
    X.right = gtree.LeafNode()

    Y = gtree.BranchNode('A', 0.5, None, None)
    Y.left = gtree.LeafNode()
    Y.right = gtree.LeafNode()

    assert X.structure_matches(Y)


def test_softmax():
    generation = [{'name': 'A', 'loss_testing': 10},
                  {'name': 'B', 'loss_testing': 20},
                  {'name': 'C', 'loss_testing': 30}]

    probs = gtree.softmax(np.array([1.0 / t['loss_testing'] for t in generation]))

    print probs, probs.dtype
    assert list([0.34641195, 0.32951724, 0.3240708]) == approx(list(probs))


def test_greedy_tree_builder():
    # Create test features and target
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       'B': [10, 20, 50, 30, 40, 50, 60, 50, 70, 90, 100, 110]}, dtype=np.float32)
    target = pd.Series([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32)

    # Do a greedy fit to produce a leaf map
    tree, leaf_map = gtree.train_greedy_tree(df, target, loss='error_rate')

    predictions = dict(tree.predict(df, leaf_map))

    print predictions

    # print leaf_map

    # Apply the leaf map
    print gtree.calculate_leaf_map(tree, df, target)


def test_random_tree():

    # Flakey :sadpanda:

    # Create test features and target
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       'B': [10, 20, 50, 30, 40, 50, 60, 50, 70, 90, 100, 110]}, dtype=np.float32)
    target = pd.Series([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32)

    best_info, history = gtree.train_random_trees(df, target,
                                             loss='cross_entropy',
                                             leaf_prediction='mean')
    # Do a greedy fit to produce a leaf map
    best_tree = best_info['tree']
    best_tree.prn()

    assert isinstance(best_info, dict)
    assert isinstance(best_info['tree'], gtree.BranchNode)


def test_evolution():
    # Create test features and target
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       'B': [10, 20, 50, 30, 40, 50, 60, 50, 70, 90, 100, 110]}, dtype=np.float32)
    target = pd.Series([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32)

    # Do a greedy fit to produce a leaf map
    best, history = gtree.evolve(df, target, loss='error_rate', leaf_prediction='mean',
                                 num_generations=4, num_seed_trees=3)

    assert isinstance(best, dict)
    assert isinstance(best['tree'], gtree.BranchNode)