import pandas as pd

import gtree

import numpy as np


def leaf_count_fn(val):
    return lambda df: pd.Series([val for _ in range(len(df))], index=df.index)


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
    leaf_map = {hash(t.left): leaf_count_fn('LEFT'),
                hash(t.right): leaf_count_fn('RIGHT')}

    assert {'baz': 'LEFT', 'foo': 'LEFT', 'bar': 'RIGHT'} == dict(t.predict(data, leaf_map))


def test_leaf_building():
    # Create test features and target
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       'B': [10, 20, 50, 30, 40, 50, 60, 50, 70, 90, 100, 110]})
    target = pd.Series([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0])

    # Do a greedy fit to produce a leaf map
    tree, leaf_map = gtree.train_greedy_tree(df, target, loss_fn=gtree.error_rate_loss)

    predictions = dict(tree.predict(df, leaf_map))

    print predictions

    # print leaf_map

    # Apply the leaf map
    print gtree.calculate_leaf_map(tree, df, target)


def test_tree_equality():
    X = gtree.BranchNode('A', 0.5, None, None)
    X.left = gtree.LeafNode()
    X.right = gtree.LeafNode()

    Y = gtree.BranchNode('A', 0.5, None, None)
    Y.left = gtree.LeafNode()
    Y.right = gtree.LeafNode()

    assert X.structure_matches(Y)


def test_softmax_choice():
    generation = [{'name': 'A', 'loss_testing': 10},
                  {'name': 'B', 'loss_testing': 20},
                  {'name': 'C', 'loss_testing': 30}]

    probs = gtree.softmax(np.array([1.0/t['loss_testing'] for t in generation]))

    print probs
    print probs.sum()

    np.random.choice(generation,
                     2,
                     p=gtree.softmax(np.array([1.0/t['loss_testing'] for t in generation])))
