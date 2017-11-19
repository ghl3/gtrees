
import numpy as np
cimport numpy as np

#ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float32 FLOAT_t         # Type of y, sample_weight

cpdef FLOAT_t combineLosses(
    FLOAT_t lossLeft, FLOAT_t weightLeft,
    FLOAT_t lossRight, FLOAT_t weightRight)

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