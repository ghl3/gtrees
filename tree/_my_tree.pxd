
import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float32 DOUBLE_t         # Type of y, sample_weight


cdef class LeafMapper:

  cpdef int init(self,
                np.ndarray[DTYPE_t, ndim=2] X,
                np.ndarray[DOUBLE_t, ndim=1, mode="c"] y) except -1

  cpdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] predict(self, np.ndarray[DTYPE_t, ndim=2] X)
