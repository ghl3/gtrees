# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf
cdef double NEG_INFINITY = np.NINF


cpdef foo(FLOAT_t a):
    return a*a


cpdef FLOAT_t combineLosses(
    FLOAT_t lossLeft, FLOAT_t weightLeft,
    FLOAT_t lossRight, FLOAT_t weightRight):

        if weightLeft + weightRight == 0:
            return 0.0
        else:
            return (lossLeft * weightLeft + lossRight * weightRight) / (weightLeft + weightRight)


cdef class LeafMapper:

    def __cinit__(self):
        pass

    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] predict(self, float[:,:] X):
        pass


cdef class MeanLeafMapper(LeafMapper):

    cdef double good_rate

    def __cinit__(self, double good_rate):
        self.good_rate = good_rate

    cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] predict(self, float[:,:] X):
        return np.full(len(X),
                       self.good_rate,
                       dtype='float32')


cdef class LeafMapperBuilder:

    def __cinit__(self):
        pass


    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef LeafMapper build(self,
                           float[:,:] X,
                           float[:] y):
        pass


cdef class MeanLeafMapperBuilder(LeafMapperBuilder):

    cpdef LeafMapper build(self,
                           float[:,:] X,
                           float[:] y):

        if len(y) > 0:

            goodRate = 0.0
            for i in range(len(y)):
                if y[i] == 1.0:
                    goodRate += 1
                goodRate /= (<float> len(y))

            return MeanLeafMapper(goodRate)
        else:
            return MeanLeafMapper(0.0)



#cdef class LogitMapperBuilder(LeafMapperBuilder):


#    cpdef LeafMapper build(self,
#                   np.ndarray[DTYPE_t, ndim=2] X,
#                   np.ndarray[DOUBLE_t, ndim=1, mode="c"] y):

#        if len(y) > 0:

#            return MeanLeafMapper(np.mean(y))
#        else:
#            return MeanLeafMapper(0)



cdef class LossFunction:
    def __cinit__(self):
        pass

    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef FLOAT_t loss(self,
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

    cpdef FLOAT_t loss(self,
                       float[:] truth,
                       float[:] predicted):

        cdef float entropy = 0.0
        cdef float count = 0.0

        if len(truth) == 0.0:
            return 0.0

        assert len(truth) == len(predicted)

        cdef float loss = 0.0
        cdef float mn = 0.00001
        cdef float mx = 0.99999
        cdef float pred = 0.0

        for i in range(len(truth)):

            pred = predicted[i]

            if pred > mx:
                pred = mx

            if pred < mn:
                pred = mn

            loss += (-1.0 * truth[i] * log(pred) - (1.0 - truth[i]) * log(1.0 - pred))

        return loss / (<float> len(truth))

        #TODO: Is this garbage collected?
        #pred = np.clip(predicted, 0.00001, .99999)

        #return ).mean()


cdef class ErrorRateLoss(LossFunction):
    """
    Loss is given by the fraction of incorrect classifications
    """

    cdef FLOAT_t _threshold

    def __cinit__(self, FLOAT_t threshold=0.5):
        self._threshold = threshold

    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef FLOAT_t loss(self,
                        float[:] truth,
                        float[:] predicted):

        assert len(truth) == len(predicted)

        #cdef int numCorrect = 0
        #cdef int numIncorrect = 0

        if len(truth) == 0:
            return 0.0

        cdef float loss = 0.0

        #cdef mn = 0.00001
        #cdef mx = 0.99999

        for i in range(len(truth)):

            if predicted[i] > self._threshold and truth[i] == 0:
                loss += 1
            elif predicted[i] < self._threshold and truth[i] == 1:
                loss += 1

        return loss / (<float> len(truth))

        #return (np.where(predicted > self._threshold, 1.0, 0.0) == truth).mean()

        #pred = np.clip(predicted, 0.00001, .99999)

        #return (-1.0 * truth * np.log(pred) - (1.0 - truth) * np.log(1.0 - pred)).mean()


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

        #def sort_by_col(fs, t, idx):
        cdef order = np.argsort(X[:, varIdx])

        # Copy the data to an array and re-order it
        #Create copied, sorted versions of the input features and targets
        ## Use 'advanced indexing' to force a copy
        cdef float[:,:] XX = np.asarray(X)[order,]
        cdef float[:] YY = np.asarray(Y)[order, ]

        cdef FLOAT_t best_loss = INFINITY
        cdef FLOAT_t best_split = NEG_INFINITY

        cdef FLOAT_t split_value = NEG_INFINITY

        #cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] PRED_left = None
        #cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] PRED_right = None

        #cdef double[:] PRED_left =

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
