from __future__ import division

# mylib.py
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
            predict_fn = leaf_score_map[leaf_hash]
            predictions = predict_fn(df_leaf)
            results.append(predictions)

        return pd.concat(results).loc[df.index]


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


class LeafNode(Node):
    def __init__(self):
        pass

    def find_leaves(self, df):
        return pd.Series(hash(self), index=df.index)

    def prn(self, indent=None):
        indent = indent or 0

        for _ in range(indent):
            print '\t',
        print "Leaf({})\n".format(hash(self))


def _get_split_candidates(srs, threshold=100):
    if len(srs) < threshold:
        return list(srs.values)
    else:
        return list(pd.qcut(srs, threshold, labels=False, retbins=True))[1]


def _single_variable_best_split(df, var, target, loss_fn, leaf_prediction_builder, candidates=None):
    # Convention:
    # Left is BAD
    # Right is GOOD

    srs = df[var]

    if candidates is None:
        candidates = _get_split_candidates(srs)

    best_loss = None
    best_split = None

    for val in candidates:

        left_idx = df.index[(srs <= val)]
        left_leaf_predict_fn = leaf_prediction_builder(df.loc[left_idx], target.loc[left_idx])
        left_predicted = left_leaf_predict_fn(df.loc[left_idx])

        right_idx = df.index[(srs > val)]
        right_leaf_predict_fn = leaf_prediction_builder(df.loc[right_idx], target.loc[right_idx])
        right_predicted = right_leaf_predict_fn(df.loc[right_idx])

        loss = loss_fn(left_predicted, target.loc[left_idx]) + loss_fn(right_predicted, target.loc[right_idx])

        if best_loss is None or loss < best_loss:
            best_split = val
            best_loss = loss

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
        # print "Loss for {} {} {}".format(var, split, loss)
        if best_loss is None or loss < best_loss:
            best_var = var
            best_split = split
            best_loss = loss

    return (best_var, best_split, best_loss)


def leaf_good_rate_split_builder(features, target):
    """
    Assume the target consists of 0, 1
    """
    if len(target) > 0:
        mean = sum(target) / len(target)
    else:
        mean = 0

    return lambda df: pd.Series([mean for _ in range(len(df))], index=df.index)


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
    predictions = predictor(df)  # .apply(predictor, axis=1)
    current_loss = loss_fn(predictions, target)

    if len(df) == 1 or (max_depth is not None and current_depth > max_depth) or (
                    min_to_split is not None and len(df) < min_to_split):
        tree_logger.info("Reached leaf node, or constraints force termination.  Returning")
        leaf = LeafNode()
        leaf_map[hash(leaf)] = leaf_prediction_builder(df, target)
        return leaf, leaf_map

    df_for_splitting = df

    if feature_sample_rate:
        df_for_splitting = df_for_splitting.sample(frac=feature_sample_rate, replace=False, axis=1)
    if row_sample_rate:
        df_for_splitting = df_for_splitting.sample(frac=row_sample_rate, replace=False, axis=0)

    var, split, loss = get_best_split(df_for_splitting, target.loc[df_for_splitting.index],
                                      loss_fn, leaf_prediction_builder, var_split_candidate_map)

    tree_logger.info("Training.  Depth {} Current Loss: {} Best Split: {} {} {}".format(
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

    return (BranchNode(var, split,
                       left_tree, right_tree),
            leaf_map)


#
#
# def train_random_tree(df, target, loss_fn,
#                       max_depth=None,
#                       min_to_split=None,
#                       leaf_map=None,
#                       feature_sample_rate=0.5,
#                       row_samle_rate=0.5,
#                       leaf_prediction_builder=leaf_good_rate_split_builder,
#                       var_split_candidate_map=None,
#                       current_depth=0):
#     """
#     Returns a tree and its leaf map
#     """
#
#     if var_split_candidate_map is None:
#         var_split_candidate_map = {var: _get_split_candidates(df[var]) for var in df.columns}
#
#     if leaf_map is None:
#         leaf_map = {}
#
#     # Calculate the loss on all the features and targes
#     predictor = leaf_prediction_builder(df, target)
#     predictions = predictor(df)  # .apply(predictor, axis=1)
#     current_loss = loss_fn(predictions, target)
#
#     if len(df) == 1 or (max_depth is not None and current_depth > max_depth) or (
#                     min_to_split is not None and len(df) < min_to_split):
#         print "Reached leaf node, or constraints force termination.  Returning"
#         leaf = LeafNode()
#         leaf_map[hash(leaf)] = leaf_prediction_builder(df, target)
#         return leaf, leaf_map
#
#
#
#     var, split, loss = get_best_split(reduced, target.loc[reduced.index], loss_fn, leaf_prediction_builder, var_split_candidate_map)
#
#     print "Training.  Depth {} Current Loss: {} Best Split: {} {} {}".format(current_depth,
#                                                                              current_loss,
#                                                                              var,
#                                                                              split,
#                                                                              loss)
#
#     if loss >= current_loss:
#         print "No split improves loss.  Returning"
#         leaf = LeafNode()
#         leaf_map[hash(leaf)] = leaf_prediction_builder(df, target)
#         return leaf, leaf_map
#
#     left_idx = df[var] <= split
#     right_idx = df[var] > split
#
#     left_tree, left_map = train_greedy_tree(df.loc[left_idx], target.loc[left_idx],
#                                             loss_fn,
#                                             max_depth=max_depth,
#                                             min_to_split=min_to_split,
#                                             leaf_map=leaf_map,
#                                             leaf_prediction_builder=leaf_prediction_builder,
#                                             var_split_candidate_map=var_split_candidate_map,
#                                             current_depth=current_depth + 1)
#
#     right_tree, right_map = train_greedy_tree(df.loc[right_idx], target.loc[right_idx],
#                                               loss_fn,
#                                               max_depth=max_depth,
#                                               min_to_split=min_to_split,
#                                               leaf_map=leaf_map,
#                                               leaf_prediction_builder=leaf_prediction_builder,
#                                               var_split_candidate_map=var_split_candidate_map,
#                                               current_depth=current_depth + 1)
#
#     leaf_map.update(left_map)
#     leaf_map.update(right_map)
#
#     return (BranchNode(var, split,
#                        left_tree, right_tree),
#             leaf_map)


def calculate_leaf_map(tree, df, target, leaf_prediction_builder=leaf_good_rate_split_builder):
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
        return [tree.left] + [tree.right] + get_all_nodes(tree.left) + get_all_nodes(tree.right)
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


def replace_node(tree, to_replace, replace_with):
    if tree == to_replace:
        return replace_with

    elif isinstance(tree, BranchNode):
        if tree.left == to_replace:
            tree.left = replace_with
            return tree
        elif tree.right == to_replace:
            tree.right = replace_with
            return tree
        else:
            replace_node(tree.left, to_replace, replace_with)
            replace_node(tree.right, to_replace, replace_with)
            return tree

    else:
        return tree


def mate(mother, father):
    """
    Create a child tree from two parent trees.
    
    We do this with the following algorithm:
    
    - Pick a node randomly in the mother tree
    - Pick a node randomly in the father tree
    - Replace the node in the mother tree
    """

    child = clone(mother)

    to_replace = random_node(child)
    replace_with = clone(random_node(father))

    return replace_node(child, to_replace, replace_with)


def cut(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def accuracy_loss(predicted, truth, threshold=0.5):
    if len(truth) == 0:
        return 0.0
    else:
        return len(truth) - np.sum((predicted >= threshold) == truth)


def cross_entropy_loss(predicted, truth):
    if len(truth) == 0:
        return 0.0
    else:
        return -1 * truth * np.log(predicted) - (1.0 - truth) * np.log(1.0 - predicted)


def evolve(df, target,
           loss_fn,
           max_depth=None,
           min_to_split=None,
           leaf_prediction_builder=leaf_good_rate_split_builder,
           num_generations=10,
           alphas_per_generation=20,
           betas_per_generation=20,
           num_parents=5,
           num_children=5):
    df_train = df.sample(frac=0.7, replace=False, axis=0)
    target_train = target.loc[df_train.index]

    df_test = df[~df.index.isin(df_train.index)]
    target_test = target.loc[df_test.index]

    # Create and cache the possible splits
    var_split_candidate_map = {var: _get_split_candidates(df[var]) for var in df.columns}

    generations = []

    older_generation = []

    for gen_idx in range(num_generations):

        trees = list(older_generation)

        # Create the alpha of this generation
        # Alphas are trees that are greedily trained with a sample
        # of the rows in the dataset
        for i in range(alphas_per_generation):
            evolution_logger.debug("Growing Alpha: {} of {}".format(i+1, alphas_per_generation))
            df_alpha = df_train.sample(frac=0.5, replace=False, axis=0)
            tree, _ = train_greedy_tree(
                df=df_alpha, target=target_train.loc[df_alpha.index],
                loss_fn=loss_fn,
                max_depth=max_depth,
                min_to_split=min_to_split,
                leaf_prediction_builder=leaf_prediction_builder,
                var_split_candidate_map=var_split_candidate_map)
            trees.append(tree)

        # Create the betas
        for i in range(betas_per_generation):
            evolution_logger.debug("Growing Betea: {} of {}".format(i+1, betas_per_generation))
            tree, _ = train_greedy_tree(
                df=df_train, target=target_train,
                loss_fn=loss_fn,
                max_depth=max_depth,
                min_to_split=min_to_split,
                leaf_prediction_builder=leaf_prediction_builder,
                feature_sample_rate=0.5,
                row_sample_rate=0.5,
                var_split_candidate_map=var_split_candidate_map)
            trees.append(tree)

        # For each tree shape, calculate the leaf performance
        # on the full training data

        trees_and_leaf_map = [(tree, calculate_leaf_map(tree, df_train, target_train, leaf_prediction_builder))
                              for tree in trees]

        # Calculate the loss on the generation
        trees_and_losses = sorted_trees_and_losses(trees_and_leaf_map, df_test, target_test, loss_fn)

        # trees_and_losses = [(tree, loss_fn(tree.predict(df_test, leaf_map), target_test))
        #                    for tree, leaf_map in trees_and_leaf_map]
        best_tree, loss_hold_out = trees_and_losses[0]  # = min([tree for tree, loss in trees_and_losses])

        loss_training =  sorted_trees_and_losses(trees_and_leaf_map, df_train, target_train, loss_fn)[0][1]

        # best_loss = min([loss for tree, loss in trees_and_losses])

        # Get the best of the generation to become parentts
        parents = [tree for tree, loss in trees_and_losses[:num_parents]]

        # Create the children
        children = []
        for _ in range(num_children):
            mother, father = random.sample(parents, 2)
            children.append(mate(mother, father))

        older_generation = parents + children

        evolution_logger.info("Generation {} Training Loss: {} Hold Out Loss {}\n".format(gen_idx, loss_training, loss_hold_out))

        generations.append({'loss_hold_out': loss_hold_out,
                            'loss_training': loss_training,
                            'generation': older_generation})

    return best_tree, generations


def sorted_trees_and_losses(trees_and_leaf_map, df, targets, loss_fn):
    # Calculate the loss on the generation
    trees_and_losses = [(tree, loss_fn(tree.predict(df, leaf_map), targets))
                        for tree, leaf_map in trees_and_leaf_map]

    return sorted(trees_and_losses, key=lambda x: x[1])

#    best_tree = min([tree for tree, loss in trees_and_losses])

#    best_loss = min([loss for tree, loss in trees_and_losses])


# def sort_by_loss(trees_and_leaf_map, loss_fn, df, target):
#    losses = []
#
#    for (tree, leaf_map) in trees_and_leaf_map:
#        predictions = tree.predict(df, leaf_map)
#        loss = loss_fn(predictions, target)
#        losses.append((tree, loss))

#    return
