# gtrees: Genetic Trees in Python

gtrees is a package for building decision tree classifiers using evolutionary methods.  gtrees allows for more customization in tree structure and fit strategies than normal tree growing methods. 

A common use case of gtrees is to build small, shallow trees whose leaf nodes are used to feed data into downstream models.  In this case, the tree can be interpreted as a segmentation that optimally separates the data before training individual models within each segment.


## Concepts

The gtrees package uses a unique structure to represent trees.  gtrees separates a built tree into two parts: it's node structure, which determines what leaf a row is associated with, and a function per leaf that makes the prediction for rows associated with that leaf.  Predicting the value of a row requires first traversing the tree with the values of that row to find the leaf node it is associated with and then  finding the leaf node AND using the map to lookup the value.

This allows one to configure different prediction functions associated with tree leaves.  Most trees use the mean of the values in the leaf for future predictions.  gtrees allows one to specify even more complicated models, such as regressions.

The loss function optimized by the tree is configurable, as is the leaf
Unlike many decision tree algorithms, gtrees allows the user to customize the loss function that the tree attempts to optimize.


## Terms

### tree

A Tree is an object that takes input data and determines what leaf it ends up in. Unlike many tree implementations, the Tree itself doesn't store data about the value of a leaf. That data must b stored externally.

### loss_fn
A loss_fn is a function that takes data rows, the predicted targets for those rows, and the actual targets for those rows, and returns a single value that determines the "LOSS" or "COST" of that prediction (lower cost/loss is better)

```
def loss_fn(predicted_targets, actual_targets) -> float
```
A loss function must be additive (so, one should not apply a mean as a part of it)


### leaf\_prediction\_fn
A leaf\_prediction\_fn is a function which takes the features and actual targets that end up in a leaf and returns a Series of the predictions for each row ending up in that leaf. It is typically a constant function whose value is either the mean good rate in that leaf (among the actual targets) or the median target, but can be anything else

```
def leaf_prediction_fn(features) -> pd.Series
```
### leaf\_prediction\_builder
A leaf\_prediction\_builder is a function which takes the features and actual targets that end up in a TRANING leaf and returns a leaf_prediction_fn. This leaf\_prediction\_fn is used to predict the value of testing rows that end up in the same leaf.

```
def leaf_prediction_builder(features, actual_targets) -> leaf_prediction_fn
```

### leaf\_prediction\_map
A leaf\_prediction\_map is a map of leaf ids (eg their hash) to the leaf\_prediction\_fn for that leaf. One can only use a tree to score data if one has a leaf\_prediction\_map. This design allows on to use the same tree as a subset of another tree without having their leaf values become entangled.
