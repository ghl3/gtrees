
import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float32 DOUBLE_t         # Type of y, sample_weight


cpdef foo(DOUBLE_t a)


cpdef DOUBLE_t combineLosses(
    DOUBLE_t lossLeft, DOUBLE_t weightLeft,
    DOUBLE_t lossRight, DOUBLE_t weightRight)


cdef class LeafMapper:
  cpdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] predict(self, np.ndarray[DTYPE_t, ndim=2] X)


cdef class LeafMapperBuilder:
  cpdef LeafMapper build(self,
                         np.ndarray[DTYPE_t, ndim=2] X,
                         np.ndarray[DOUBLE_t, ndim=1, mode="c"] y)


cdef class LossFunction:
    cpdef DOUBLE_t loss(self,
                        np.ndarray[DOUBLE_t, ndim=1, mode="c"] truth,
                        np.ndarray[DOUBLE_t, ndim=1, mode="c"] predicted)


cpdef tuple getBestSplit(
    np.ndarray[DOUBLE_t, ndim=2] X,
    int varIdx,
    np.ndarray[DOUBLE_t, ndim=1, mode="c"] Y,
    LossFunction lossFunction,
    LeafMapperBuilder leafMapperBuilder,
    set splitCandidates)