
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
def combined_counting_opt(cnp.ndarray[DTYPE_UINT32_t, ndim=3] args_max,
                          cnp.ndarray[DTYPE_UINT32_t, ndim=3] args_min, 
                          cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] optims_max, 
                          cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] optims_min, 
                          unsigned int kernels_per_group):

    cdef unsigned int num_examples = optims_max.shape[0]
    cdef unsigned int num_groups   = optims_max.shape[1]
    cdef unsigned int num_samples  = optims_max.shape[2]
    cdef unsigned int idx_max, idx_min, ex, gr, s
    cdef float optim_max

    cdef cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] feats = np.zeros([num_examples, num_groups, 2*kernels_per_group], dtype=DTYPE_FLOAT32)

    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=22):
            for gr in range(num_groups):
                for s in range(num_samples):
                    optim_max = optims_max[ex,gr,s]
                    idx_max   = args_max[ex,gr,s]
                    idx_min   = args_min[ex,gr,s]

                    # Soft-Max
                    feats[ex,gr,idx_max*2+0] += optim_max

                    # Hard-Min
                    feats[ex,gr,idx_min*2+1] += 1
    
    return feats