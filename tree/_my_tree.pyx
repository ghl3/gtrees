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

#ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight


cdef class LeafMapper:

    def __cinit__(self):
        pass


    def __dealloc__(self):
        """Destructor."""
        pass

    cpdef int init(self,
                   np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=1, mode="c"] y) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples

        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        pass

    cpdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        pass


cdef class MeanLeafMapper(LeafMapper):

    cdef double good_rate

    def __cinit__(self):
        self.good_rate = -1

    cpdef int init(self,
                   np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=1, mode="c"] y) except -1:

        self.good_rate = np.mean(y)


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

