from __future__ import division

import math
import logging

import random
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

tree_logger = logging.getLogger('tree')
evolution_logger = logging.getLogger('evolution')

import tree._my_tree


class Node(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def find_leaves(self, df):
        raise NotImplemented()

    @abstractmethod
    def prn(self, indent=None):
        raise NotImplemented()

    def predict(self, df, leaf_score_map):
        # A pd Series of leaf-hash values
        # for each row
        leaf_for_row = self.find_leaves(df)

        results = []

        for leaf_hash, df_leaf in df.groupby(leaf_for_row):
            predict_fn = leaf_score_map.get(leaf_hash, tree._my_tree.MeanLeafMapper(0.5))
            # The predict_fn return a np array,
            # so we have to convert it back to a pd Series
            predictions = pd.Series(predict_fn.predict(df_leaf.values), index=df_leaf.index)
            results.append(predictions)

        return pd.concat(results).loc[df.index]

    @abstractmethod
    def structure_matches(self, other):
        raise NotImplemented()


class BranchNode(Node):
    def __init__(self, var_name, split, left, right):
        self.var_name = var_name
        self.split = split
        self.left = left
        self.right = right

    def find_leaves(self, df):
        idx_left = (df[self.var_name] < self.split)
        idx_right = (df[self.var_name] >= self.split)

        left_leaves = self.left.find_leaves(df.loc[idx_left])
        right_leaves = self.right.find_leaves(df.loc[idx_right])
        return pd.concat([left_leaves, right_leaves]).loc[df.index]

    def prn(self, indent=None):

        indent = indent or 0

        if self.left:
            self.left.prn(indent + 1)

        for _ in range(indent):
            print '\t',
        print "{} {:.5f}".format(self.var_name, self.split)

        if self.right:
            self.right.prn(indent + 1)

    def __eq__(self, o):
        return isinstance(o, BranchNode) \
               and self.var_name == o.var_name \
               and self.split == o.split \
               and self.left == o.left \
               and self.right == o.right

    def __hash__(self):
        return hash((self.var_name, self.split, self.left, self.right))

    def structure_matches(self, o):
        return isinstance(o, BranchNode) \
               and self.var_name == o.var_name \
               and self.split == o.split \
               and self.left.structure_matches(o.left) \
               and self.right.structure_matches(o.right)


class LeafNode(Node):
    def __init__(self):
        self._code = random.random()

    def find_leaves(self, df):
        return pd.Series(hash(self), index=df.index)

    def prn(self, indent=None):
        indent = indent or 0

        for _ in range(indent):
            print '\t',
        print "<Leaf {}>".format(hash(self))

    def __eq__(self, o):
        return isinstance(o, LeafNode) and self._code == o._code

    def __hash__(self):
        return hash(self._code)

    def structure_matches(self, other):
        return isinstance(other, LeafNode)

#
# Loss Functions
#


def _get_leaf_prediction_builder(leaf_prediction):
    if isinstance(leaf_prediction, tree._my_tree.LeafMapperBuilder):
        return leaf_prediction

    elif leaf_prediction == 'mean':
        return tree._my_tree.MeanLeafMapperBuilder()

    elif leaf_prediction == 'logit':
        return tree._my_tree.LogitMapperBuilder()

    else:
        raise Exception()


def get_leaf_predictor(X, y, type):
    builder = _get_leaf_prediction_builder(type)
    return builder.build(X, y)


def _get_loss_function(loss):
    if isinstance(loss, tree._my_tree.LossFunction):
        return loss

    elif loss == 'cross_entropy':
        return tree._my_tree.CrossEntropyLoss()

    elif loss == 'error_rate':
        return tree._my_tree.ErrorRateLoss()

    elif loss == 'random':
        return tree._my_tree.RandomLoss()

    else:
        raise Exception()


def loss(truth, predicted, type):
    loss_fn = _get_loss_function(type)
    return loss_fn.loss(truth.values, predicted.values)


#
# Tree Manipulation
#


def get_all_nodes(tree):
    if isinstance(tree, LeafNode):
        return [tree]
    elif isinstance(tree, BranchNode):
        return [tree] + [tree.left] + [tree.right] + get_all_nodes(tree.left) + get_all_nodes(tree.right)
    else:
        raise ValueError()


def clone(tree):
    if isinstance(tree, LeafNode):
        return LeafNode()
    elif isinstance(tree, BranchNode):
        return BranchNode(tree.var_name, tree.split, clone(tree.left), clone(tree.right))
    else:
        raise ValueError()


def random_node(tree):
    return random.choice(get_all_nodes(tree))


def random_branch_node(tree):
    if isinstance(tree, LeafNode):
        raise ValueError()

    while True:
        node = random.choice(get_all_nodes(tree))
        if isinstance(node, BranchNode):
            return node


def replace_branch_split(tree, to_replace, replace_with):
    """
    Takes a tree and a node in that tree to replace
    and a node to replace it with.
    Replace that node in a shallow or "in-place" way by
    replacing that node's variable and threshold but keeping
    it's children the same

    This mutates the tree in-place and returns the updated version
    """

    if isinstance(to_replace, LeafNode):
        raise ValueError("Cannot call replace_branch_split on a LeafNode")

    tree = clone(tree)

    for node in get_all_nodes(tree):
        if node == to_replace:
            node.var_name = replace_with.var_name
            node.split = replace_with.split
            return tree

    return tree


def replace_node(tree, to_replace, replace_with):
    """
    Takes a tree and a node in that tree to replace
    and a node to replace it with.
    Replace that node in a deep way by removing the
    original node fro the tree and replacing it with
    the replacement node (including it's children).

    This may even replace leaf nodes, so it may alter
    the structure of the tree.

    This may mutates the tree in-place and always returns the updated version
    """

    # Handle the case where we replace the root
    if tree == to_replace:
        return clone(replace_with)

    tree = clone(tree)

    # Otherwise, find it in the tree
    for node in get_all_nodes(tree):

        if isinstance(node, LeafNode):
            continue

        elif isinstance(node, BranchNode):

            if node.left == to_replace:
                node.left = replace_with
                return tree

            elif node.right == to_replace:
                node.right = replace_with
                return tree

            else:
                pass

    return tree


def mate(mother, father):
    """
    Create a child tree from two parent trees.

    We do this with the following algorithm:

    - Pick a node randomly in the mother tree
    - Pick a node randomly in the father tree
    - Replace the node in the mother tree
    """

    num_genes = random.randint(1, 4)

    child = mother

    for gene in range(num_genes):

        if isinstance(child, LeafNode):
            return child

        # Do we do a full subtree replacement
        # or do we modify a branch node?
        if not isinstance(father, LeafNode) and random.choice([True, False]):
            to_replace = random_branch_node(child)
            replace_with = random_branch_node(father)
            child = replace_branch_split(child, to_replace, replace_with)
        else:
            to_replace = random_node(child)
            replace_with = random_node(father)
            child = replace_node(child, to_replace, replace_with)

    return child


def mutate(tree, df, target, loss_fn, leaf_prediction_builder, mutation_rate=1.0):
    # Can't mutate leaf nodes
    if isinstance(tree, LeafNode):
        return tree

    tree = clone(tree)

    # Pick the number of genes to mutate based
    # on a poisson distribution
    num_genes_to_mutate = np.random.poisson(mutation_rate, 1)[0]

    num_genes_mutated = 0

    while num_genes_mutated < num_genes_to_mutate:
        # How do we mutate?
        # - pick a node at random to mutate
        # - Pick a feature to mutate it to
        # - Make a greedy laf split
        to_mutate = random_branch_node(tree)
        new_feature = random.choice(df.columns)

        # Pick either a greedy split
        # or a random split
        if random.choice([True, False], ):
            split_val, _ = _single_variable_best_split(df, new_feature, target, loss_fn, leaf_prediction_builder)
        else:
            split_val = df[new_feature].sample(n=1).iloc[0]

        # Do the mutation
        # We do in-place because this is a child
        to_mutate.var_name = new_feature
        to_mutate.split = split_val
        num_genes_mutated += 1

    return tree


def prune(tree, max_depth=None, current_depth=0):
    """
    Return a (possibly cloned) version of the tree
    that is pruned to respect the max depth given.
    """

    if isinstance(tree, LeafNode):
        return tree

    elif max_depth is None:
        return tree

    else:

        tree = clone(tree)

        # Force all children to be leaf node
        if current_depth == max_depth - 1:
            tree.left = LeafNode()
            tree.right = LeafNode()
            return tree
        else:
            tree.left = prune(tree.left, max_depth, current_depth + 1)
            tree.right = prune(tree.right, max_depth, current_depth + 1)
            return tree


#
# Training
#


def _get_split_candidates(srs, threshold=100):
    if len(srs) < threshold:
        return list(srs)
    else:
        skip = len(srs) // 100
        return list(srs[::skip])


def _single_variable_best_split(df, var, target, loss='cross_entropy', leaf_prediction='mean', candidates=None):
    """
    Takes a DataFrame of features, a variable name,
    and a target Series, and finds a value of the
    input variable name that best splits the data
    according to the targets.

    Returns the best value to split at and the
    value of the loss function when splitting there.

    Internally, wraps a cython function
    """

    leaf_prediction_builder = _get_leaf_prediction_builder(leaf_prediction)
    loss_fn = _get_loss_function(loss)

    np_features = df.astype(np.float32).values
    np_targets = target.astype(np.float32).values

    var_idx = list(df.columns).index(var)

    if candidates is None:
        candidates = set(_get_split_candidates(df[var].astype(np.float32)))
    else:
        candidates = set(candidates)

    return tree._my_tree.getBestSplit(
        np_features,
        var_idx,
        np_targets,
        loss_fn,
        leaf_prediction_builder,
        candidates)


def get_best_split(df, target, loss='cross_entropy', leaf_prediction='mean'):
    """
    Takes a DataFrame of features and a target Series,
    and find the value of the input variable name
    that best splits the data according to the targets.

    Returns the best variable to split on, the best value
    of that variable to split at, and the loss when splitting
    on that variable at that value.

    Internally, wraps a cython function
    """

    leaf_prediction_builder = _get_leaf_prediction_builder(leaf_prediction)
    loss_fn = _get_loss_function(loss)

    best_var = None
    best_split = None
    best_loss = None

    for var in df.columns:
        split, loss = _single_variable_best_split(df, var, target, loss_fn, leaf_prediction_builder)
        if best_loss is None or loss < best_loss:
            best_var = var
            best_split = split
            best_loss = loss

    return best_var, best_split, best_loss


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_greedy_tree(df,
                      target,
                      loss='cross_entropy',
                      leaf_prediction='mean',
                      max_depth=None,
                      min_to_split=None,
                      leaf_map=None,
                      var_split_candidate_map=None,
                      feature_sample_rate=None,
                      row_sample_rate=None,
                      current_depth=0):
    """
    Returns a tree and its leaf map
    """

    assert target.dtype == np.float32
    for dtype in df.dtypes:
        assert dtype == np.float32

    if leaf_map is None:
        leaf_map = {}

    loss_fn = _get_loss_function(loss)
    leaf_prediction_builder = _get_leaf_prediction_builder(leaf_prediction)

    predictor = leaf_prediction_builder.build(df.values, target.values)
    predictions = predictor.predict(df.values)
    current_loss = loss_fn.loss(predictions, target.values)

    if len(df) <= 1 or (max_depth is not None and current_depth >= max_depth) or (
                    min_to_split is not None and len(df) < min_to_split):
        tree_logger.info("Reached leaf node, or constraints force termination.  Returning")
        leaf = LeafNode()
        leaf_map[hash(leaf)] = predictor
        return leaf, leaf_map

    df_for_splitting = sample(df, row_frac=row_sample_rate, col_frac=feature_sample_rate)

    var, split, loss = get_best_split(df_for_splitting,
                                      target.loc[df_for_splitting.index],
                                      loss_fn,
                                      leaf_prediction_builder)

    tree_logger.info("Training.  Depth {} Current Loss: {:.4f} Best Split: {} {:.4f} {:.4f}".format(
        current_depth,
        current_loss,
        var,
        split,
        loss))

    if loss >= current_loss:
        tree_logger.info("No split improves loss.  Returning")
        leaf = LeafNode()
        leaf_map[hash(leaf)] = predictor
        return leaf, leaf_map

    left_criteria = df[var] < split
    right_criteria = df[var] >= split

    # Handle the (odd) case where the split
    # moves all noes to one way or another
    if left_criteria.sum() == 0 or right_criteria.sum() == 0:
        tree_logger.info("No split improves loss.  Returning")
        leaf = LeafNode()
        leaf_map[hash(leaf)] = leaf_prediction_builder.build(df, target)
        return leaf, leaf_map

    left_tree, left_map = train_greedy_tree(df[left_criteria],
                                            target[left_criteria],
                                            loss_fn,
                                            leaf_prediction_builder,
                                            max_depth=max_depth,
                                            min_to_split=min_to_split,
                                            leaf_map=leaf_map,
                                            var_split_candidate_map=var_split_candidate_map,
                                            current_depth=current_depth + 1)

    right_tree, right_map = train_greedy_tree(df[right_criteria],
                                              target[right_criteria],
                                              loss_fn,
                                              leaf_prediction_builder,
                                              max_depth=max_depth,
                                              min_to_split=min_to_split,
                                              leaf_map=leaf_map,
                                              var_split_candidate_map=var_split_candidate_map,
                                              current_depth=current_depth + 1)

    leaf_map.update(left_map)
    leaf_map.update(right_map)

    return BranchNode(var, split, left_tree, right_tree), leaf_map


def calculate_leaf_map(tree, df, target, leaf_prediction='mean'):
    """
    Takes a built tree structure and a features/target pair
    and returns a map of each leaf to the function evaluating
    the score at each leaf
    """

    leaves = tree.find_leaves(df)

    leaf_map = {}
    leaf_prediction_builder = _get_leaf_prediction_builder(leaf_prediction)

    for leaf_hash, leaf_rows in df.groupby(leaves):
        leaf_targets = target.loc[leaf_rows.index]
        leaf_map[leaf_hash] = leaf_prediction_builder.build(leaf_rows.values, leaf_targets.values)

    return leaf_map


def cut(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def train_random_trees(df,
                       target,
                       loss='cross_entropy',
                       leaf_prediction='mean',
                       max_depth=None,
                       min_to_split=None,
                       num_trees=10,
                       num_split_candidates=50):
    loss_fn = _get_loss_function(loss)
    leaf_prediction_builder = _get_leaf_prediction_builder(leaf_prediction)

    df_train = df.sample(frac=0.7, replace=False, axis=0)
    target_train = target.loc[df_train.index]

    df_test = df[~df.index.isin(df_train.index)]
    target_test = target.loc[df_test.index]

    # Create and cache the possible splits
    var_split_candidate_map = {var: _get_split_candidates(df[var], threshold=num_split_candidates) for var in
                               df.columns}

    tree_infos = []

    num_grown_trees = 0

    while num_grown_trees < num_trees:

        random_type = random.choice(['alpha', 'beta'])

        if random_type == 'alpha':

            df_alpha = sample(df_train, row_frac=0.5)
            target_alpha = target_train.loc[df_alpha.index]
            tree, _ = train_greedy_tree(
                df=df_alpha,
                target=target_alpha,
                loss=loss_fn,
                leaf_prediction=leaf_prediction_builder,
                max_depth=max_depth,
                min_to_split=min_to_split,
                var_split_candidate_map=var_split_candidate_map)

        else:
            tree, _ = train_greedy_tree(
                df=df_train,
                target=target_train,
                loss=loss_fn,
                leaf_prediction=leaf_prediction_builder,
                max_depth=max_depth,
                min_to_split=min_to_split,
                feature_sample_rate=0.5,
                row_sample_rate=0.5,
                var_split_candidate_map=var_split_candidate_map)

        num_grown_trees += 1
        tree_info = {'tree': tree,
                     'random_type': random_type}

        # Calculate the loss on the generation
        loss_info = calculate_loss(tree,
                                   leaf_prediction_builder,
                                   loss_fn,
                                   df_train, target_train,
                                   df_test, target_test)

        tree_info.update(loss_info)
        tree_infos.append(tree_info)

        tree_infos = sorted(tree_infos, key=lambda x: x['loss_testing'])
        best_result = tree_infos[0]

        evolution_logger.info("Num Trees: {} Training Loss: {:.4f} Hold Out Loss {:.4f}\n".format(
            num_grown_trees,
            best_result['loss_training'],
            best_result['loss_testing']))

    return best_result, tree_infos


def evolve(df,
           target,
           loss='cross_entropy',
           leaf_prediction='mean',
           max_depth=None,
           min_to_split=None,
           num_generations=10,
           num_survivors=10,
           num_children=50,
           num_split_candidates=50,
           num_seed_trees=5):
    loss_fn = _get_loss_function(loss)
    leaf_prediction_builder = _get_leaf_prediction_builder(leaf_prediction)

    df_train = df.sample(frac=0.7, replace=False, axis=0)
    target_train = target.loc[df_train.index]

    assert 0 == pd.isnull(target_train).sum(), "Targets may not have NULL values"

    df_test = df[~df.index.isin(df_train.index)]
    target_test = target.loc[df_test.index]

    # Create and cache the possible splits
    var_split_candidate_map = {var: _get_split_candidates(df[var], threshold=num_split_candidates) for var in
                               df.columns}

    generation_info = []

    generation = []

    # Create the alpha of this generation
    # Alphas are trees that are greedily trained with a sample
    # of the rows in the dataset
    for i in range(num_seed_trees):
        evolution_logger.debug("Growing Seed: {} of {}".format(i + 1, num_seed_trees))
        df_seed = sample(df_train, row_frac=0.5)
        target_seed = target_train.loc[df_seed.index]
        tree, _ = train_greedy_tree(
            df=df_seed,
            target=target_seed,
            loss='cross_entropy',
            leaf_prediction='mean',
            max_depth=max_depth,
            min_to_split=min_to_split,
            var_split_candidate_map=var_split_candidate_map)
        generation.append({'tree': tree,
                           'gen': 0,
                           'loss_training': None,
                           'loss_testing': None})

    for gen_idx in range(num_generations):

        # Get the training data for this generation
        evolution_logger.debug("Resplitting the data")
        df_gen = df_train.sample(frac=0.7, replace=False, axis=0)
        target_gen = target.loc[df_gen.index]

        # Create the children for this generation
        evolution_logger.debug("Mating to create {} children".format(num_children))

        children = []

        # Pick parents inversely proportionally to their loss
        # This is probably overly complicated...
        losses = np.array([t['loss_testing'] if t['loss_testing'] else 1.0 for t in generation])
        probs = softmax(1.0 - losses / np.mean(losses))
        for _ in range(num_children):
            mother, father = np.random.choice(generation, 2, p=probs)
            child = mate(mother['tree'], father['tree'])
            if isinstance(child, LeafNode):
                continue
            child = mutate(child, df_gen, target_gen, loss_fn, leaf_prediction_builder)
            child = prune(child, max_depth=max_depth)
            children.append({'gen': max(mother['gen'], father['gen']) + 1,
                             'tree': child})

        generation = list(generation) + children

        generation = ensure_diversity(generation)

        # Calculate the leaf weights for this generation
        # and evaluate on the hold-out set
        evolution_logger.debug("Calculating loss functions for generation of size: {}".format(len(generation)))
        losses = calculate_losses(generation,
                                  leaf_prediction_builder,
                                  loss_fn,
                                  df_gen, target_gen,
                                  df_test, target_test)

        # Update the losses in the generation
        for tree, losses in zip(generation, losses):
            tree.update(losses)
            #tree['loss_training'] = losses['loss_training']
            #tree['loss_testing'] = losses['loss_testing']

        # Sort the trees to find the best tree
        next_generation = sorted(generation, key=lambda x: x['loss_testing'])[:num_survivors]

        best_result = next_generation[0]

        evolution_logger.debug(
            "Surviving Generation: {}".format(", ".join(['{}:{:.4f}'.format(r['gen'], r['loss_testing'])
                                                         for r in next_generation])))

        evolution_logger.info(
            "Generation {} Training Loss: {:.4f} Hold Out Loss {:.4f}\n".format(
                gen_idx,
                best_result['loss_training'],
                best_result['loss_testing'])
        )

        generation = next_generation

        generation_info.append({'best_of_generation': best_result,
                                'generation': generation})

    return best_result, generation_info


def calculate_loss(tree,
                   leaf_prediction_builder,
                   loss_fn,
                   df_train, target_train,
                   df_test, target_test):

    # Calculate the leaf map on the training data and apply to training/testing
    leaf_map = calculate_leaf_map(tree, df_train, target_train, leaf_prediction_builder)
    loss_training = loss_fn.loss(tree.predict(df_train, leaf_map).values, target_train.values)
    loss_testing = loss_fn.loss(tree.predict(df_test, leaf_map).values, target_test.values)

    return {
        'loss_training': loss_training,
        'loss_testing': loss_testing,
        'leaf_map': leaf_map
    }


def calculate_losses(tree_infos,
                     leaf_prediction_builder,
                     loss_fn,
                     df_train, target_train,
                     df_test, target_test):
    results = []

    for info in tree_infos:
        loss_info = calculate_loss(info['tree'], leaf_prediction_builder, loss_fn,
                                   df_train, target_train,
                                   df_test, target_test)
        results.append(loss_info)

    return results


def ensure_diversity(trees):
    res = []
    for tree in trees:

        # If a tree matches structurally an existing tree,
        # we skip it

        is_diverse = True

        for r in res:
            if tree['tree'].structure_matches(r['tree']):
                is_diverse = False
                break

        if is_diverse:
            res.append(tree)

    return res


def sample(df, row_frac=None, col_frac=None):
    if len(df) == 0:
        return df
    elif row_frac is None and col_frac is None:
        return df
    elif row_frac is None:
        ncols = int(math.floor(len(df.columns) * col_frac))
        cols = random.sample(df.columns, ncols)
        return df.loc[:, cols]
    elif col_frac is None:
        nrows = int(math.floor(len(df) * row_frac))
        rows = random.sample(df.index, nrows)
        return df.loc[rows, :]
    else:
        ncols = int(math.floor(len(df.columns) * col_frac))
        cols = random.sample(df.columns, ncols)
        nrows = int(math.floor(len(df) * row_frac))
        rows = random.sample(df.index, nrows)
        return df.loc[rows, cols]
