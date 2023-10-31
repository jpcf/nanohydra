
import numpy as np
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE_UINT32 = np.uint32
ctypedef cnp.uint32_t DTYPE_UINT32_t


@cython.boundscheck(False)
@cython.wraparound(False)
def hard_counting_opt(cnp.ndarray[DTYPE_UINT32_t, ndim=3] max_idxs, unsigned int kernels_per_group):
    cdef unsigned int num_examples = max_idxs.shape[0]
    cdef unsigned int num_groups   = max_idxs.shape[1]
    cdef unsigned int num_samples  = max_idxs.shape[2]
    cdef unsigned int idx, ex, gr, s

    cdef cnp.ndarray[DTYPE_UINT32_t, ndim=3] feats = np.zeros([num_examples, num_groups, kernels_per_group], dtype=DTYPE_UINT32)

    for ex in range(num_examples):
        for gr in range(num_groups):
            for s in range(num_samples):
                idx = max_idxs[ex,gr,s]
                feats[ex,gr,idx] += 1
    
    return feats