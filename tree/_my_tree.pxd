
# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 FLOAT_t         # Type of y, sample_weight

cdef float combineLosses(
    float lossLeft, float weightLeft,
    float lossRight, float weightRight) nogil

cdef class LeafMapper:
  cpdef np.ndarray[FLOAT_t, ndim=1, mode="c"] predict(self, float[:,:] X)

cdef class LeafMapperBuilder:
  cpdef LeafMapper build(self,
                         float[:,:] X,
                         float[:] y)

cdef class LossFunction:
    cpdef FLOAT_t loss(self,
                        float[:] truth,
                        float[:] predicted)

cpdef tuple getBestSplit(
    float[:,:] X,
    int varIdx,
    float[:] Y,
    LossFunction lossFunction,
    LeafMapperBuilder leafMapperBuilder,
    set splitCandidates)