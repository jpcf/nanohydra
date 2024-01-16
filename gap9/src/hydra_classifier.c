#include "../include/hydra.h"


void hydra_classifier(Hydra* hydra) {
    // This code is meant to be run in a x64 platform, just for validating
    // the processing chain agains the model. It will be replaced by a GAP9-optimized 
    // matrix-vector multiplier
    for(int c=0; c < hydra->N_classes; c++) {
        for(int f=0; f < hydra->len_feat_vec; f++) {
            hydra->classf_scores[c] += hydra->featVec[f] * hydra->classf_weights[c][f];
        }
        hydra->classf_scores[c] += hydra->classf_bias[c];
    }
}