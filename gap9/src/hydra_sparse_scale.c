#include "../include/hydra.h"

void hydra_sparse_scale(Hydra *hydra) {

    for(int f=0; f < hydra->len_feat_vec; f++) {

        // Under-clip to zero, skip normalization if feature is zero
        hydra->featVec[f] = (hydra->featVec[f] < 0 ? 0 : hydra->featVec[f]);
        
        if(hydra->featVec[f] > 0) {
            hydra->featVec[f] = hydra->featVec[f] - hydra->featMean[f];

            if(hydra->featVec[f] < 0) {
                // Most values are positive, but arithmetic shift of negative numbers 
                // is not equivalent to division by powers of two, since it does not round to zero.
                for(int s = 0; s < hydra->featStd[f]; s++) {
                    hydra->featVec[f] = (hydra->featVec[f]) / 2;
                }
            }
            else {
                hydra->featVec[f] = (hydra->featVec[f]) >> hydra->featStd[f];
            }
        }
    }
}