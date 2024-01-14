#include "../include/hydra.h"

void hydra_forward(int16_t  **inX,
                   int16_t  **inX_diff,
                   int16_t ***inW, 
                   int16_t   *featVec, 
                   Hydra     *hydra) {

    uint8_t dil_idx;
    uint8_t diff_idx;
    uint8_t chan;
    uint16_t dil;

    // Iterate through the work chunks, for each dil/diff combination
    for (dil_idx = 0; dil_idx < hydra->N_dil; dil_idx++) {
        dil = generate_dilation_val(dil_idx);
        for (diff_idx = 0; diff_idx < hydra->N_diff; diff_idx++) {
            for (chan = 0; chan < hydra->N_chan; chan++) {
                // Parallelization point for OMP Tasks. Since Chan and Dil are highly variable from
                // problem to problem, this is a more generic solution for both Single and Multichannel problems.
                if(diff_idx == 0) {
                    hydra_convolve(inX[chan],      
                                   inW, 
                                   &(featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]), 
                                   dil, 
                                   hydra,
                                   diff_idx);
                }
                else {
                    hydra_convolve(inX_diff[chan], 
                                   inW, 
                                   &(featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]), 
                                   dil, 
                                   hydra,
                                   diff_idx);
                }
            }
        }
    }
}