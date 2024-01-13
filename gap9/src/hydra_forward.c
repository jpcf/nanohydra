#include "../include/hydra_utils.h"
#include "../include/hydra_convolve.h"
#include <stdint.h>

void hydra_forward(int16_t  **inX,
                   int16_t  **inX_diff,
                   int16_t ***inW, 
                   int16_t   *featVec, 
                   uint16_t lenX,  uint16_t lenW, 
                   uint16_t H,     uint16_t K, 
                   uint8_t N_dil,  uint8_t  N_diff, 
                   uint8_t N_chan, uint8_t N_feats) {

    uint8_t dil_idx;
    uint8_t diff_idx;
    uint8_t chan;
    uint16_t xi;
    uint16_t dil;

    // Iterate through the work chunks, for each dil/diff combination
    for (dil_idx = 0; dil_idx < N_dil; dil_idx++) {
        dil = generate_dilation_val(dil_idx);
        for (diff_idx = 0; diff_idx < N_diff; diff_idx++) {
            for (chan = 0; chan < N_chan; chan++) {
                // Parallelization point for OMP Tasks. Since Chan and Dil are highly variable from
                // problem to problem, this is a more generic solution for both Single and Multichannel problems.
                if(diff_idx == 0) {
                    hydra_convolve(inX[chan],      
                                   inW, 
                                   &(featVec[chan*(K*H*N_feats) + diff_idx*(K*H*N_chan*N_feats) + dil_idx*(K*H*N_chan*N_feats*N_diff)]), 
                                   dil, 
                                   lenX, 
                                   lenW, 
                                   padding_len(lenW, N_dil),
                                   H, 
                                   K, 
                                   N_feats);
                }
                else {
                    hydra_convolve(inX_diff[chan], 
                                   inW, 
                                   &(featVec[chan*(K*H*N_feats) + diff_idx*(K*H*N_chan*N_feats) + dil_idx*(K*H*N_chan*N_feats*N_diff)]), 
                                   dil, 
                                   lenX-1, 
                                   lenW, 
                                   padding_len(lenW, N_dil),
                                   H, 
                                   K, 
                                   N_feats);
                }
            }
        }
    }
}