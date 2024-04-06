import numpy as np
import multiprocessing
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE_X = np.int16
ctypedef cnp.int16_t DTYPE_X_t

DTYPE_W = np.int16
ctypedef cnp.int16_t DTYPE_W_t

DTYPE_Y = np.int32
ctypedef cnp.int32_t DTYPE_Y_t

@cython.boundscheck(False)
@cython.wraparound(False)
def conv1d_opt_x_int16_w_b1(cnp.ndarray[DTYPE_X_t, ndim=2] x, cnp.ndarray[DTYPE_W_t, ndim=3] w, unsigned int dilation):

    cdef unsigned int num_examples = x.shape[0]
    cdef unsigned int xlen   = x.shape[1]
    cdef unsigned int wlen   = w.shape[2]
    cdef unsigned int H = w.shape[0]
    cdef unsigned int K = w.shape[1]
    cdef unsigned int xpad_len = (9//2)*(dilation+1)+1

    cdef unsigned int h,k,xi,wi,ex

    cdef cnp.ndarray[DTYPE_Y_t, ndim=4] Y     = np.zeros([num_examples, H, K, xlen], dtype=DTYPE_Y)
    cdef cnp.ndarray[DTYPE_X_t, ndim=2] x_dil = np.zeros([num_examples, xlen+xpad_len*2], dtype=DTYPE_X)

    x_dil[:,xpad_len:xlen+xpad_len] = x[:,:]

    # Work-sharing construct must start here, since np.take uses gil.
    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=24):
            for h in range(H):
                for k in range(K):
                    for xi in range(xlen):
                        for wi in range(wlen):
                            Y[ex, h, k, xi] += x_dil[ex,xi+xpad_len+(wi-4)*(dilation+1)]*w[h,k,wi]

    return Y
