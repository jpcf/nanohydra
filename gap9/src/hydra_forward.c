#include "../include/hydra.h"

#if defined (TARGET_GAP9) && defined (PARALLELIZE)
#else
void hydra_forward(Hydra *hydra) {

    uint8_t dil_idx;
    uint8_t diff_idx;
    uint8_t chan;
    uint16_t dil;

    // Iterate through the work chunks, for each dil/diff combination
    for (dil_idx = 0; dil_idx < hydra->N_dil; dil_idx++) {
        dil = generate_dilation_val(dil_idx);
        for (diff_idx = 0; diff_idx < hydra->N_diff; diff_idx++) {
            for (chan = 0; chan < hydra->N_chan; chan++) {
                if(diff_idx == 0) {
                    hydra_convolve(hydra->inX[chan],      
                                   hydra->inW, 
                                   &(hydra->featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]), 
                                   dil, 
                                   hydra,
                                   diff_idx);
                }
                else {
                    hydra_convolve(hydra->inX_diff[chan], 
                                   hydra->inW, 
                                   &(hydra->featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]), 
                                   dil, 
                                   hydra,
                                   diff_idx);
                }
            }
        }
    }
}
#endif