#include "../include/hydra.h"


void hydra_classifier(Hydra* hydra) {
    #if defined (TARGET_GAP9) && defined (VECTORIZE)
    v4s featVec;
    v4s classf_weights;
    int8_t *pFeatVec = (int8_t*) hydra->featVec;
    #endif

    for(int c=0; c < hydra->N_classes; c++) {
        #if defined (VECTORIZE)
        for(int f=0; f < hydra->len_feat_vec; f+=4) {
        #else
        for(int f=0; f < hydra->len_feat_vec; f+=1) {
        #endif
            #if defined (TARGET_GAP9) && defined (VECTORIZE)
            featVec                  = __builtin_pulp_pack4(hydra->featVec[f], hydra->featVec[f+1], hydra->featVec[f+2], hydra->featVec[f+3]);
            classf_weights           = *((v4s*) &hydra->classf_weights[c][f]);
            hydra->classf_scores[c]  = __builtin_pulp_sdotsp4(featVec, classf_weights, hydra->classf_scores[c]);
            #else
            hydra->classf_scores[c] += hydra->featVec[f] * hydra->classf_weights[c][f];
            #endif
        }
        hydra->classf_scores[c] += hydra->classf_bias[c];
    }
}