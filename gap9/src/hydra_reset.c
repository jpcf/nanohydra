#include <stdint.h>

void hydra_reset(int16_t *featVec, 
                 uint16_t lenX, uint16_t lenW, 
                 uint16_t H,    uint16_t K, 
                 uint8_t N_dil, uint8_t  N_diff, 
                 uint8_t N_chan, uint8_t N_feats) {

    uint16_t i;

    // Reset the feature vector, parallelization point for OMP Parallel For
    for(i=0; i < H*K*N_dil*N_diff*N_feats*N_chan; i++) {
        featVec[i] = 0;
    }
}