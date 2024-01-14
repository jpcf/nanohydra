#include "../include/hydra.h"

void hydra_reset(Hydra *hydra) {

    uint16_t i;

    // Reset the feature vector, parallelization point for OMP Parallel For
    for(i=0; i < hydra->len_feat_vec; i++) {
        hydra->featVec[i] = 0;
    }
}