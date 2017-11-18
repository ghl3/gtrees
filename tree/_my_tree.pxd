
import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float32 DOUBLE_t         # Type of y, sample_weight





cdef class LeafMapper:

#  cpdef int init(self,
#                np.ndarray[DTYPE_t, ndim=2] X,
#                np.ndarray[DOUBLE_t, ndim=1, mode="c"] y) except -1

  cpdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] predict(self, np.ndarray[DTYPE_t, ndim=2] X)



cdef class LeafMapperBuilder:

  cpdef LeafMapper build(self,
                         np.ndarray[DTYPE_t, ndim=2] X,
                         np.ndarray[DOUBLE_t, ndim=1, mode="c"] y)


cdef class LossFunction:

    cpdef DOUBLE_t loss(self,
                        np.ndarray[DOUBLE_t, ndim=1, mode="c"] truth,
                        np.ndarray[DOUBLE_t, ndim=1, mode="c"] predicted)

    cpdef DOUBLE_t combineLosses(self,
                                 DOUBLE_t lossLeft, DOUBLE_t weightLeft,
                                 DOUBLE_t lossRight, DOUBLE_t weightRight)


cdef class SpitFinder:

    cpdef tuple getBestSplit(self,
                                int varIdx,
                                set splitCandidates,
                                np.ndarray[DOUBLE_t, ndim=2] X,
                                np.ndarray[DOUBLE_t, ndim=1, mode="c"] Y,
                                LeafMapperBuilder leafMapperBuilder,
                                LossFunction lossFunction)