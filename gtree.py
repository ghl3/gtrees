from __future__ import division

import math
import random
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod


class Node(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def find_leaves(self, df):
        raise NotImplemented()
        
    @abstractmethod
    def prn(self, indent=None):
        raise NotImplemented()
            
    def predict(self, df, leaf_score_map):
        return self.find_leaves(df).map(lambda l: leaf_score_map[l])
        

class BranchNode(Node):
    
    def __init__(self, var_name, split, left, right):
        self.var_name = var_name
        self.split = split
        self.left = left
        self.right = right
        
    def find_leaves(self, df):
        idx_left = (df[self.var_name] <= self.split)        
        left_leaves = self.left.find_leaves(df[idx_left])            
        right_leaves = self.right.find_leaves(df[~idx_left])                
        return pd.concat([left_leaves, right_leaves]).loc[df.index]
    
    def prn(self, indent=None):
        
        indent = indent or 0
        
        if self.left:
            self.left.prn(indent+1)
        
        for _ in range(indent):
            print '\t',
        print "{} {}\n".format(self.var_name, self.split)
            
        if self.right:
            self.right.prn(indent+1)
                    
    
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


def _single_variable_best_split(df, var, target, loss_fn):
    
    # Convention:
    # Left is BAD
    # Right is GOOD
    
    srs = df[var]
    
    if len(srs) < 100:
        candidates = srs.values
    else:
        _, candidates = pd.qcut(srs, 100, labels=False, retbins=True)
    
    best_loss = None
    best_split = None
    
    for val in candidates:
        left_idx = (srs <= val)
                
        left_truth = target[left_idx]
        left_predicted = [left_truth.mean() for _ in left_truth]
                
        right_truth = target[~left_idx]
        right_predicted = [right_truth.mean() for _ in right_truth]
                
        loss = loss_fn(df.loc[left_truth.index], left_truth, left_predicted) + loss_fn(df.loc[right_truth.index], right_truth, right_predicted)
        
        if best_loss is None or loss < best_loss:
            best_split = val
            best_loss = loss
            
    return best_split, best_loss
 

def get_best_split(df, target, loss_fn):
    # Return:
    # (var, split, loss)
    
    best_var = None
    best_split = None
    best_loss = None
    
    for var in df.columns:
        split, loss = _single_variable_best_split(df, var, target, loss_fn)
        if best_loss is None or loss < best_loss:
            best_var = var
            best_split = split
            best_loss = loss
            
    return (best_var, best_split, best_loss)    


def leaf_good_rate(features, target):
    counts = dict(target.value_counts())
    num_good = counts.get(1, 0)
    num_bad = counts.get(0, 0)
    return num_good / (num_good + num_bad)
#    return num_good / (num_bad + num_good)
    

def train_greedy_tree(df, target, loss_fn,
                      max_depth=None,
                      min_to_split=None,
                      leaf_map=None,
                      leaf_value_fn=leaf_good_rate):
    """
    Returns a tree and its leaf map
    """
    
    if leaf_map is None:
        leaf_map = {}
    
    current_loss = loss_fn(df, target, [leaf_good_rate(df, target) for _ in target])
    
    if len(df) == 1 or (max_depth is not None and max_depth <= 0) or (min_to_split is not None and len(df) < min_to_split):
        leaf = LeafNode()
        leaf_map[hash(leaf)] = leaf_value_fn(df, target)
        return leaf, leaf_map
    
    var, split, loss = get_best_split(df, target, loss_fn)

    if loss >= current_loss:
        leaf = LeafNode()
        leaf_map[hash(leaf)] = leaf_value_fn(df, target)
        return leaf, leaf_map
    
    left_idx = df[var] <= split
    
    left_tree, left_map = train_greedy_tree(df[left_idx], target[left_idx],
                                            loss_fn,
                                            max_depth = max_depth-1 if max_depth else None,
                                            min_to_split=min_to_split,
                                            leaf_map=leaf_map,
                                            leaf_value_fn=leaf_value_fn)
                                  
    right_tree, right_map = train_greedy_tree(df[~left_idx], target[~left_idx],
                                              loss_fn,
                                              max_depth = max_depth-1 if max_depth else None,
                                              min_to_split=min_to_split,
                                              leaf_map=leaf_map,
                                              leaf_value_fn=leaf_value_fn)
    
    leaf_map.update(left_map)
    leaf_map.update(right_map)
    
    return (BranchNode(var, split,
                      left_tree, right_tree),
            leaf_map)


def calculate_leaf_map(tree, df, target, leaf_value_fn=leaf_good_rate):
    
    leaf_map = {}
    
    leaves = tree.find_leaves(df)
    
    for leaf_hash, leaf_rows in df.groupby(leaves):
        
        
        leaf_map[leaf_hash] = leaf_value_fn(leaf_rows,
                                            target.loc[leaf_rows.index])
    
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

def cross_entropy(df, truth, predicted, threshold=0.5):
    
    if len(predicted) == 0:
        return 0
    
    loss = 0
    
    for (p, t) in zip(predicted, truth):
    
        # Correct prediction
        if (p >= threshold and t==1) or (p < threshold and t==0):
            loss += math.log(cut(p, 0.001, 0.999))
        else:
            loss += math.log(cut(1.0-p, 0.001, 0.999))
        
    return loss

