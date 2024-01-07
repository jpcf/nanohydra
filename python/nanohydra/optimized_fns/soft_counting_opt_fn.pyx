
import numpy as np
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE_FLOAT32 = np.float32
ctypedef cnp.float32_t DTYPE_FLOAT32_t

DTYPE_UINT32 = np.uint32
ctypedef cnp.uint32_t DTYPE_UINT32_t

@cython.boundscheck(False)
@cython.wraparound(False)
def soft_counting_opt(cnp.ndarray[DTYPE_UINT32_t, ndim=3] args, cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] optims, unsigned int kernels_per_group):
    cdef unsigned int num_examples = optims.shape[0]
    cdef unsigned int num_groups   = optims.shape[1]
    cdef unsigned int num_samples  = optims.shape[2]
    cdef unsigned int idx , ex, gr, s
    cdef float optim

    cdef cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] feats = np.zeros([num_examples, num_groups, kernels_per_group], dtype=DTYPE_FLOAT32)

    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=22):
            for gr in range(num_groups):
                for s in range(num_samples):
                    optim = optims[ex,gr,s]
                    idx   = args[ex,gr,s]
                    feats[ex,gr,idx] += optim
    
    return feats