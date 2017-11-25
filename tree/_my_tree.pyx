# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False
# cython: linetrace=False

from libc.math cimport log
from libc.stdlib cimport rand, RAND_MAX

import numpy as np
cimport numpy as np
np.import_array()
from numpy.math cimport INFINITY

# Logit stuff
from scipy.special import expit
from sklearn.svm.base import _fit_liblinear

cdef float combineLosses(
    float lossLeft, float weightLeft,
    float lossRight, float weightRight) nogil:

        if weightLeft + weightRight == 0:
            return 0.0
        else:
            return (lossLeft * weightLeft + lossRight * weightRight) / (weightLeft + weightRight)


cdef class LeafMapper:

    cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] predict(self, float[:,:] X):
        pass


cdef class LeafMapperBuilder:

    cpdef LeafMapper build(self,
                           float[:,:] X,
                           float[:] y):
        pass


cdef class MeanLeafMapper(LeafMapper):

    cdef double good_rate

    def __cinit__(self, double good_rate):
        self.good_rate = good_rate

    cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] predict(self, float[:,:] X):
        return np.full(len(X),
                       self.good_rate,
                       dtype='float32')


cdef class MeanLeafMapperBuilder(LeafMapperBuilder):

    cpdef LeafMapper build(self,
                           float[:,:] X,
                           float[:] y):

        cdef float goodRate = 0.0

        if len(y) > 0:
            for i in range(len(y)):
                if y[i] == 1.0:
                    goodRate += 1.0

            goodRate /= (<float> len(y))
            return MeanLeafMapper(goodRate)
        else:
            return MeanLeafMapper(0.0)


cdef class LogitMapper(LeafMapper):

    cdef float[:] coeficients
    cdef float intercept

    def __cinit__(self, float[:] coeficients, float intercept):
        self.coeficients = coeficients
        self.intercept = intercept

    cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] predict(self, float[:,:] X):
        return expit(np.dot(X, self.coeficients.T) + self.intercept)
        #return expit(0.0 + self.intercept)
        #return 0.0

    cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] get_coeficients(self):
        return np.asarray(self.coeficients)

    cpdef float get_intercept(self):
        return self.intercept


cdef class LogitMapperBuilder(LeafMapperBuilder):

    cpdef LeafMapper build(self,
                           float[:,:] X,
                           float[:] y):


        target_vals = list(set(y))

        if len(target_vals) == 2:

             coefs, intercept, _ = _fit_liblinear(
                np.asarray(X, dtype=np.float64),
                np.asarray(np.ravel(y), dtype=np.float64),
                penalty='l1', C=1.0,
                fit_intercept=True, intercept_scaling=1.0,
                class_weight=None, sample_weight=None,
                dual=False, verbose=False,
                max_iter=50, tol=1e-3, random_state=None)

             return LogitMapper(np.asarray(coefs[0], dtype=np.float32), intercept)
        elif len(target_vals) == 1:
            return MeanLeafMapper(target_vals[0])
        elif len(target_vals) == 0:
            return MeanLeafMapper(0)
        else:
            raise Exception()


cdef class LossFunction:

    cpdef float loss(self,
                       float[:] truth,
                       float[:] predicted):
        pass


cdef class CrossEntropyLoss(LossFunction):
    """Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cpdef float loss(self,
                       float[:] truth,
                       float[:] predicted):

        cdef float entropy = 0.0
        cdef float count = 0.0

        if truth.shape[0] == 0:
            return 0.0

        assert truth.shape[0] == predicted.shape[0]

        cdef float loss = 0.0
        cdef float mn = 0.00001
        cdef float mx = 0.99999
        cdef float pred = 0.0

        cdef int i = 0

        for i in range(truth.shape[0]):

            pred = predicted[i]

            if pred > mx:
                pred = mx

            elif pred < mn:
                pred = mn

            if truth[i] >= 0.5:
                loss += -1.0 * log(pred)
            else:
                loss += -1.0 * log(1.0 - pred)

        return loss / (<float> truth.shape[0])


cdef class ErrorRateLoss(LossFunction):
    """
    Loss is given by the fraction of incorrect classifications
    """

    cdef float _threshold

    def __cinit__(self, float threshold=0.5):
        self._threshold = threshold

    cpdef float loss(self,
                        float[:] truth,
                        float[:] predicted):

        assert truth.shape[0] == predicted.shape[0]

        if truth.shape[0] == 0:
            return 0.0

        cdef float loss = 0.0

        for i in range(truth.shape[0]):

            if predicted[i] > self._threshold and truth[i] == 0:
                loss += 1
            elif predicted[i] < self._threshold and truth[i] == 1:
                loss += 1

        return loss / (<float> truth.shape[0])


cdef class RandomLoss(LossFunction):
    cpdef float loss(self,
                        float[:] truth,
                        float[:] predicted):
        return rand()/(<float> RAND_MAX)



cpdef tuple getBestSplit(
    float[:,:] X,
    int varIdx,
    float[:] Y,
    LossFunction lossFunction,
    LeafMapperBuilder leafMapperBuilder,
    set splitCandidates):

        # Sort X and y by the variable in question
        # Iterate along the variable
        # Calcualte the loss at each split

        cdef long[:] order = np.argsort(X[:, varIdx])

        # Copy the data to an array and re-order it
        cdef float[:,:] XX = np.asarray(X)[order,]
        cdef float[:] YY = np.asarray(Y)[order, ]

        cdef float best_loss = INFINITY
        cdef float best_split = -1*INFINITY

        cdef float split_value = -1* INFINITY

        cdef size_t splitIdx = 0

        for splitIdx in range(X.shape[0]):

            # Consider new spit points at the BEGINNING
            # of a new feature value (new value goes to
            # the right, meaning we do: < and >=)
            if XX[splitIdx, varIdx] == split_value:
                continue
            else:
                split_value = XX[splitIdx, varIdx]

            if split_value not in splitCandidates:
                continue

            # These should be views, not copies
            X_left = XX[0:splitIdx, :]
            Y_left = YY[0:splitIdx]
            left_leaf_predict_fn = leafMapperBuilder.build(X_left, Y_left)
            left_loss = lossFunction.loss(Y_left, left_leaf_predict_fn.predict(X_left))

            # These should be views, not copies
            X_right = XX[splitIdx:len(XX), :]
            Y_right = YY[splitIdx:len(YY)]
            right_leaf_predict_fn = leafMapperBuilder.build(X_right, Y_right)
            #cdef double[:] PRED_right =
            right_loss = lossFunction.loss(Y_right, right_leaf_predict_fn.predict(X_right))

            avg_loss = combineLosses(left_loss, <double> len(X_left),
                                     right_loss, <double> len(X_right))

            if avg_loss < best_loss:
                best_split = split_value
                best_loss = avg_loss

        return (best_split, best_loss)

