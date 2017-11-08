from __future__ import division

import math
import logging

import random
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

tree_logger = logging.getLogger('tree')
evolution_logger = logging.getLogger('evolution')


class Node(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def find_leaves(self, df):
        raise NotImplemented()

    @abstractmethod
    def prn(self, indent=None):
        raise NotImplemented()

    def predict(self, df, leaf_score_map):
        leaf_for_row = self.find_leaves(df)

        results = []

        for leaf_hash, df_leaf in df.groupby(leaf_for_row):
            predict_fn = leaf_score_map.get(leaf_hash,
                                            lambda df: pd.Series([0.5 for _ in range(len(df))], index=df.index))
            predictions = predict_fn(df_leaf)
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
        idx_left = (df[self.var_name] <= self.split)
        left_leaves = self.left.find_leaves(df.loc[idx_left])
        right_leaves = self.right.find_leaves(df.loc[~idx_left])
        return pd.concat([left_leaves, right_leaves]).loc[df.index]

    def prn(self, indent=None):

        indent = indent or 0

        if self.left:
            self.left.prn(indent + 1)

        for _ in range(indent):
            print '\t',
        print "{} {}\n".format(self.var_name, self.split)

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
        print "Leaf(id={})\n".format(self._code)

    def __eq__(self, o):
        return isinstance(o, LeafNode) and self._code == o._code

    def __hash__(self):
        return hash(self._code)

    def structure_matches(self, other):
        return isinstance(other, LeafNode)


def _get_split_candidates(srs, threshold=100):
    if len(srs) < threshold:
        return list(srs.values)
    else:
        return list(pd.qcut(srs, threshold, labels=False, retbins=True))[1]


def _single_variable_best_split(df, var, target, loss_fn, leaf_prediction_builder, candidates=None):
    # Convention:
    # Left is BAD
    # Right is GOOD

    # TODO: Optimize me!
    # Try: df.reindex_axis(index, copy=False)
    # or:  df.reindex(index=['a', 'b'], copy=False)
    # or even: df._reindex_axes(axes={'index':df.index, 'columns': df.columns},
    # copy=False, level=None, limit=None, tolerance=None, method=None, fill_value=None)
    # From generic.py: 2594

    srs = df[var]

    if candidates is None:
        candidates = _get_split_candidates(srs)

    if len(srs) <= len(candidates):
        candidates = srs.values

    best_loss = None
    best_split = None

    for val in candidates:

        left_idx = df.index[(srs <= val)]
        df_left = df.reindex_axis(left_idx, copy=False)  # df.loc[left_idx]
        target_left = target.loc[left_idx]
        left_leaf_predict_fn = leaf_prediction_builder(df_left, target_left)
        left_predicted = left_leaf_predict_fn(df_left)

        right_idx = df.index[(srs > val)]
        df_right = df.reindex_axis(right_idx, copy=False)  # df.loc[right_idx]
        target_right = target.loc[right_idx]
        right_leaf_predict_fn = leaf_prediction_builder(df_right, target_right)
        right_predicted = right_leaf_predict_fn(df_right)

        left_loss = loss_fn(left_predicted, target_left)
        right_loss = loss_fn(right_predicted, target_right)
        avg_loss = (left_loss * len(left_idx) + right_loss * len(right_idx)) / (len(df))

        if best_loss is None or avg_loss < best_loss:
            best_split = val
            best_loss = avg_loss

    return best_split, best_loss


def get_best_split(df, target, loss_fn, leaf_prediction_builder, var_split_candidate_map=None):
    # Return:
    # (var, split, loss)

    best_var = None
    best_split = None
    best_loss = None

    for var in df.columns:

        if var_split_candidate_map is not None:
            candidate = var_split_candidate_map.get(var, None)
        else:
            candidate = None

        split, loss = _single_variable_best_split(df, var, target, loss_fn, leaf_prediction_builder, candidate)
        if best_loss is None or loss < best_loss:
            best_var = var
            best_split = split
            best_loss = loss

    return best_var, best_split, best_loss


def leaf_good_rate_split_builder(features, target):
    """
    Assume the target consists of 0, 1
    """
    if len(target) > 0:
        mean = sum(target) / len(target)
    else:
        mean = 0

    return lambda df: pd.Series([mean for _ in range(len(df))], index=df.index)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_greedy_tree(df, target, loss_fn,
                      max_depth=None,
                      min_to_split=None,
                      leaf_map=None,
                      leaf_prediction_builder=leaf_good_rate_split_builder,
                      var_split_candidate_map=None,
                      feature_sample_rate=None,
                      row_sample_rate=None,
                      current_depth=0):
    """
    Returns a tree and its leaf map
    """

    if var_split_candidate_map is None:
        var_split_candidate_map = {var: _get_split_candidates(df[var]) for var in df.columns}

    if leaf_map is None:
        leaf_map = {}

    predictor = leaf_prediction_builder(df, target)
    predictions = predictor(df)
    current_loss = loss_fn(predictions, target)

    if len(df) <= 1 or (max_depth is not None and current_depth >= max_depth) or (
                    min_to_split is not None and len(df) < min_to_split):
        tree_logger.info("Reached leaf node, or constraints force termination.  Returning")
        leaf = LeafNode()
        leaf_map[hash(leaf)] = leaf_prediction_builder(df, target)
        return leaf, leaf_map

    df_for_splitting = sample(df, row_frac=row_sample_rate, col_frac=feature_sample_rate)

    var, split, loss = get_best_split(df_for_splitting, target.loc[df_for_splitting.index],
                                      loss_fn, leaf_prediction_builder, var_split_candidate_map)

    tree_logger.info("Training.  Depth {} Current Loss: {:.4f} Best Split: {} {:.4f} {:.4f}".format(
        current_depth,
        current_loss,
        var,
        split,
        loss))

    if loss >= current_loss:
        tree_logger.info("No split improves loss.  Returning")
        leaf = LeafNode()
        leaf_map[hash(leaf)] = leaf_prediction_builder(df, target)
        return leaf, leaf_map

    left_idx = df[var] <= split
    right_idx = df[var] > split

    left_tree, left_map = train_greedy_tree(df.loc[left_idx], target.loc[left_idx],
                                            loss_fn,
                                            max_depth=max_depth,
                                            min_to_split=min_to_split,
                                            leaf_map=leaf_map,
                                            leaf_prediction_builder=leaf_prediction_builder,
                                            var_split_candidate_map=var_split_candidate_map,
                                            current_depth=current_depth + 1)

    right_tree, right_map = train_greedy_tree(df.loc[right_idx], target.loc[right_idx],
                                              loss_fn,
                                              max_depth=max_depth,
                                              min_to_split=min_to_split,
                                              leaf_map=leaf_map,
                                              leaf_prediction_builder=leaf_prediction_builder,
                                              var_split_candidate_map=var_split_candidate_map,
                                              current_depth=current_depth + 1)

    leaf_map.update(left_map)
    leaf_map.update(right_map)

    return BranchNode(var, split, left_tree, right_tree), leaf_map


def calculate_leaf_map(tree, df, target, leaf_prediction_builder=leaf_good_rate_split_builder):
    """
    Takes a built tree structure and a features/target pair
    and returns a map of each leaf to the function evaluating
    the score at each leaf
    """

    leaf_map = {}

    leaves = tree.find_leaves(df)

    for leaf_hash, leaf_rows in df.groupby(leaves):
        leaf_targets = target.loc[leaf_rows.index]
        leaf_map[leaf_hash] = leaf_prediction_builder(leaf_rows, leaf_targets)

    return leaf_map


def get_all_nodes(tree):
    if isinstance(tree, LeafNode):
        return [tree]
    elif isinstance(tree, BranchNode):
        return [tree] + [tree.left] + [tree.right] + get_all_nodes(tree.left) + get_all_nodes(tree.right)
    else:
        raise ValueError()


def clone(tree):
    if isinstance(tree, LeafNode):
        c = LeafNode()
        c._code = tree._code
        return c
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

        if random.choice([True, False]):
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
        if random.choice([True, False]):
            split_val, _ = _single_variable_best_split(df, new_feature, target, loss_fn, leaf_prediction_builder)
        else:
            split_val = df[new_feature].sample(n=1).iloc[0]

        # Do the mutation
        # We do in-place because this is a child
        to_mutate.var_name = new_feature
        to_mutate.split = split_val

        # Do we mutate WHICH VARAIBLE
        # if random.choice([True, False]):
        #
        #    to_mutate.split = features[to_mutate.var_name].sample(n=1).iloc[0]
        # else:
        #    to_mutate = random_branch_node(tree)
        #    to_mutate.var_name =

        num_genes_mutated += 1

    return tree


def prune(tree, max_depth=None, current_depth=0):
    """
    Return a (possibly cloned) version of the tree
    that is pruned to respect the max depth given.
    """

    if isinstance(tree, LeafNode):
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


def cut(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def error_rate_loss(predicted, truth, threshold=0.5):
    if len(truth) == 0:
        return 0.0
    else:
        return 1.0 - np.mean((predicted >= threshold) == truth)


def cross_entropy_loss(predicted, truth):
    if len(truth) == 0:
        return 0.0
    else:
        return (-1.0 * truth * np.log(predicted) - (1.0 - truth) * np.log(1.0 - predicted)).mean()


def train_random_trees(df, target,
                       loss_fn,
                       max_depth=None,
                       min_to_split=None,
                       leaf_prediction_builder=leaf_good_rate_split_builder,
                       num_trees=10,
                       num_split_candidates=50):
    df_train = df.sample(frac=0.7, replace=False, axis=0)
    target_train = target.loc[df_train.index]

    df_test = df[~df.index.isin(df_train.index)]
    target_test = target.loc[df_test.index]

    # Create and cache the possible splits
    var_split_candidate_map = {var: _get_split_candidates(df[var], threshold=num_split_candidates) for var in
                               df.columns}

    trees = []

    num_grown_trees = 0

    while num_grown_trees < num_trees:

        if random.choice([True, False]):

            df_alpha = sample(df_train, row_frac=0.5)
            target_alpha = target_train.loc[df_alpha.index]
            tree, _ = train_greedy_tree(
                df=df_alpha, target=target_alpha,
                loss_fn=loss_fn,
                max_depth=max_depth,
                min_to_split=min_to_split,
                leaf_prediction_builder=leaf_prediction_builder,
                var_split_candidate_map=var_split_candidate_map)
            trees.append(('alpha', tree))
            num_grown_trees += 1

        else:
            tree, _ = train_greedy_tree(
                df=df_train, target=target_train,
                loss_fn=loss_fn,
                max_depth=max_depth,
                min_to_split=min_to_split,
                leaf_prediction_builder=leaf_prediction_builder,
                feature_sample_rate=0.5,
                row_sample_rate=0.5,
                var_split_candidate_map=var_split_candidate_map)
            trees.append(('beta', tree))
            num_grown_trees += 1

        # For each tree shape, calculate the leaf performance
        # on the full training data
        trees_and_leaf_map = [(type, tree, calculate_leaf_map(tree, df_train, target_train, leaf_prediction_builder))
                              for type, tree in trees]

        # Calculate the loss on the generation
        results = calculate_losses(trees, leaf_prediction_builder, loss_fn,
                                   df_train, target_train,
                                   df_test, target_test)

        best_result = results[0]

        # best_type, best_tree, loss_hold_out = \
        #    sorted_trees_and_losses(trees_and_leaf_map, df_test, target_test, loss_fn)[0]
        # _, _, loss_training = sorted_trees_and_losses(trees_and_leaf_map, df_train, target_train, loss_fn)[0]

        evolution_logger.info("Num Trees: {} Training Loss: {:.4f} Hold Out Loss {:.4f}\n".format(
            num_grown_trees,
            best_result['loss_training'],
            best_result['loss_testing']))

    return best_result


def evolve(df, target,
           loss_fn,
           max_depth=None,
           min_to_split=None,
           leaf_prediction_builder=leaf_good_rate_split_builder,
           num_generations=10,
           num_survivors=10,
           num_children=50,
           num_split_candidates=50,
           num_seed_trees=5):
    df_train = df.sample(frac=0.7, replace=False, axis=0)
    target_train = target.loc[df_train.index]

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
            df=df_seed, target=target_seed,
            loss_fn=loss_fn,
            max_depth=max_depth,
            min_to_split=min_to_split,
            leaf_prediction_builder=leaf_prediction_builder,
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
        print "Losses: {}".format(losses)
        print "Probs: {}".format(probs)
        for _ in range(num_children):

            mother, father = np.random.choice(generation, 2, p=probs)

            child = mate(mother['tree'], father['tree'])
            child = mutate(child, df_gen, target_gen, loss_fn, leaf_prediction_builder)
            child = prune(child, max_depth=max_depth)
            children.append({'gen': max(mother['gen'], father['gen']) + 1,
                             'tree': child})

        generation = list(generation) + children

        generation = ensure_diversity(generation)

        # Calculate the leaf weights for this generation
        # and evaluate on the hold-out set
        losses = calculate_losses(generation, leaf_prediction_builder, loss_fn,
                                  df_gen, target_gen,
                                  df_test, target_test)

        for tree, losses in zip(generation, losses):
            tree['loss_training'] = losses['loss_training']
            tree['loss_testing'] = losses['loss_testing']

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


def calculate_losses(trees, leaf_prediction_builder, loss_fn,
                     df_train, target_train,
                     df_test, target_test):
    results = []

    for info in trees:
        # Calculate the leaf map on the training data
        leaf_map = calculate_leaf_map(info['tree'], df_train, target_train, leaf_prediction_builder)

        loss_training = loss_fn(info['tree'].predict(df_train, leaf_map), target_train)
        loss_testing = loss_fn(info['tree'].predict(df_test, leaf_map), target_test)

        results.append({
            'loss_training': loss_training,
            'loss_testing': loss_testing})

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
