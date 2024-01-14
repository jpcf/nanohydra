#include "../include/hydra.h"

void hydra_reset(int16_t *featVec, Hydra *hydra) {

    uint16_t i;

    // Reset the feature vector, parallelization point for OMP Parallel For
    for(i=0; i < hydra->len_feat_vec; i++) {
        featVec[i] = 0;
    }
}