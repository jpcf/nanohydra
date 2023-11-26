import numpy as np
import multiprocessing
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE_X = np.float32
ctypedef cnp.float32_t DTYPE_X_t

DTYPE_W = np.int16
ctypedef cnp.int16_t DTYPE_W_t

@cython.boundscheck(False)
@cython.wraparound(False)
def conv1d_opt_x_f32_w_b1(cnp.ndarray[DTYPE_X_t, ndim=2] x, cnp.ndarray[DTYPE_W_t, ndim=3] w, unsigned int dilation):

    cdef unsigned int num_examples = x.shape[0]
    cdef unsigned int xlen   = x.shape[1]
    cdef unsigned int wlen   = w.shape[2]
    cdef unsigned int wlenD2 = int((w.shape[2]-1)/2)
    cdef unsigned int H = w.shape[0]
    cdef unsigned int K = w.shape[1]
    cdef unsigned int xdil_len = int(xlen/(dilation+1))

    cdef unsigned int h,k,xi,wi,xidx,ex

    cdef cnp.ndarray[DTYPE_X_t, ndim=4] Y = np.zeros([num_examples, H, K, xdil_len], dtype=DTYPE_X)

    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=24):
            for h in range(H):
                for k in range(K):
                    for xi in range(0, xlen, 1):
                        for wi in range(wlen):
                            xidx = (1+dilation)*(xi+wi-wlenD2)
                            if(xidx >= 0 and xidx < xlen):
                                Y[ex, h, k, xi] += x[ex,xidx]*w[h,k,wi]
    return Y
