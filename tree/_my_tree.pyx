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


#ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight


cdef class LeafMapper:

    def __cinit__(self):
        pass


    def __dealloc__(self):
        """Destructor."""
        pass

#cpdef int init(self,
#               np.ndarray[DTYPE_t, ndim=2] X,
#               np.ndarray[DOUBLE_t, ndim=1, mode="c"] y) except -1:
#    """Initialize the splitter.
#
#        Take in the input data X, the target Y, and optional sample weights.
#
#        Returns -1 in case of failure to allocate memory (and raise MemoryError)
#        or 0 otherwise.
#
#        Parameters
#        ----------
#        X : object
#            This contains the inputs. Usually it is a 2d numpy array.
#
#        y : numpy.ndarray, dtype=DOUBLE_t
#            This is the vector of targets, or true labels, for the samples
#
#        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
#            The weights of the samples, where higher weighted samples are fit
#            closer than lower weight samples. If not provided, all samples
#            are assumed to have uniform weight.
#        """
#
#        pass

    cpdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        pass





cdef class MeanLeafMapper(LeafMapper):

    cdef double good_rate

    def __cinit__(self, double good_rate):
        self.good_rate = good_rate

#    cpdef int init(self,
#                   np.ndarray[DTYPE_t, ndim=2] X,
#                   np.ndarray[DOUBLE_t, ndim=1, mode="c"] y) except -1:
#
#        self.good_rate = np.mean(y)


    cpdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        #print X.shape[0]
        #print self.good_rate
        #cdef np.ndarray[DOUBLE_t, ndim=1] leaf_vals = np.full(X.shape[0], self.good_rate, dtype='float32')
        #return leaf_vals
        #cdef int len = X.shape[0]
        #print type(X.shape[0])
        return np.full(X.shape[0], #3, #np.array([len]),
                       self.good_rate,
                       dtype='float32')






cdef class LeafMapperBuilder:

    def __cinit__(self):
        pass


    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef LeafMapper build(self,
                   np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=1, mode="c"] y):
        pass


cdef class MeanLeafMapperBuilder(LeafMapperBuilder):


    cpdef LeafMapper build(self,
                   np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=1, mode="c"] y):

        return MeanLeafMapper(np.mean(y))





cdef class LossFunction:
    def __cinit__(self):
        pass

    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef DOUBLE_t loss(self,
                        np.ndarray[DOUBLE_t, ndim=1, mode="c"] truth,
                        np.ndarray[DOUBLE_t, ndim=1, mode="c"] predicted):
        pass

    cpdef DOUBLE_t combineLosses(self,
                                 DOUBLE_t lossLeft, DOUBLE_t weightLeft,
                                 DOUBLE_t lossRight, DOUBLE_t weightRight):
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

    cpdef DOUBLE_t loss(self,
                       np.ndarray[DOUBLE_t, ndim=1, mode="c"] truth,
                       np.ndarray[DOUBLE_t, ndim=1, mode="c"] predicted):

        #cdef SIZE_t* n_classes = self.n_classes
        #cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count = 0.0
        #cdef SIZE_t k
        #cdef SIZE_t c

        assert len(truth) == len(predicted)

        #  return (-1.0 * truth * np.log(predicted) - (1.0 - truth) * np.log(1.0 - predicted)).mean()

        for i in range(len(truth)):

            pred = np.clip(predicted[i], 0.000001, .999999)

            entropy += -1.0 * truth[i] * log(pred) - (1.0 - truth[i]) * log(1.0 - pred)
            count += 1.0

            #    count_k = sum_total[c]
            #    if count_k > 0.0:
            #        count_k /= self.weighted_n_node_samples
            #        entropy -= count_k * log(count_k)

            #sum_total += self.sum_stride

        if count == 0:
            return 1.0
        else:
            return entropy / count


    cpdef DOUBLE_t combineLosses(self,
                                 DOUBLE_t lossLeft, DOUBLE_t weightLeft,
                                 DOUBLE_t lossRight, DOUBLE_t weightRight):

        if weightLeft + weightRight == 0:
            return 1.0
        else:
            return (lossLeft * weightLeft + lossRight * weightRight) / (weightLeft + weightRight)


cdef class SpitFinder:

    def __cinit__(self):
        pass

    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef tuple getBestSplit(self,
                                int varIdx,
                                np.ndarray[DOUBLE_t, ndim=2] X,
                                np.ndarray[DOUBLE_t, ndim=1, mode="c"] Y,
                                LeafMapperBuilder leafMapperBuilder,
                                LossFunction lossFunction):

        # Sort X and y by the variable in question
        # Iterate along the variable
        # Calcualte the loss at each split

        #cdef np.ndarray[DOUBLE_t, ndim=2] XX = X.copy()
        #cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] XX = Y.copy()


        #def sort_by_col(fs, t, idx):
        cdef order = np.argsort(X[:, varIdx])

        # Create copied, sorted versions of the input features and targets
        # Use 'advanced indexing' to force a copy
        cdef np.ndarray[DOUBLE_t, ndim=2] XX = X[order,]
        cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] YY = Y[order, ]

        cdef DOUBLE_t best_loss = INFINITY
        cdef DOUBLE_t best_split = NEG_INFINITY

        cdef DOUBLE_t split_value = NEG_INFINITY

        cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] PRED_left = None
        cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] PRED_right = None

        for splitIdx in range(X.shape[0]):

            # Consider new spit points at the BEGINNING
            # of a new feature value (new value goes to
            # the right, meaning we do: < and >=)
            if XX[splitIdx, varIdx] == split_value:
                continue
            else:
                split_value = XX[splitIdx, varIdx]

            X_left = XX[0:splitIdx, :]
            Y_left = YY[0:splitIdx]
            left_leaf_predict_fn = leafMapperBuilder.build(X_left, Y_left)
            PRED_left = left_leaf_predict_fn.predict(X_left)
            left_loss = lossFunction.loss(Y_left, PRED_left)

            X_right = XX[splitIdx:-1, :]
            Y_right = YY[splitIdx:-1]
            right_leaf_predict_fn = leafMapperBuilder.build(X_right, Y_right)
            PRED_right = right_leaf_predict_fn.predict(X_right)
            right_loss = lossFunction.loss(Y_right, PRED_right)

            avg_loss = lossFunction.combineLosses(left_loss, <double> len(X_left),
                                                  right_loss, <double> len(X_right))


            if avg_loss < best_loss:
                best_split = split_value
                best_loss = avg_loss

        return (best_split, best_loss)
