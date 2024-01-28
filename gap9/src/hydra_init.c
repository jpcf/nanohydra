#include "../include/hydra.h"

Hydra* hydra_init(
    uint16_t   lenX,
    uint16_t   lenW,
    uint16_t   K,
    uint16_t   G,
    uint8_t    N_dil,
    uint8_t    N_diff,
    uint8_t    N_chan, 
    uint8_t    N_feats,
    uint8_t    N_classes,
    uint8_t    conv_frac_bit_shift) {

    // Declare pointer to hydra struct
    Hydra *hydra;
    
    // Initialize the Hydra struct
    hydra            = (Hydra*) malloc(sizeof(Hydra));
    hydra->lenX      = lenX;
    hydra->lenW      = lenW;
    hydra->lenXpad   = padding_len(hydra->lenW,generate_dilation_val(hydra->N_dil))+100; // ToDO: For high dil, we need extra padding.
    hydra->K         = K;
    hydra->G         = G;
    hydra->N_dil     = N_dil;
    hydra->N_feats   = N_feats;
    hydra->N_diff    = N_diff;
    hydra->N_chan    = N_chan;
    hydra->N_classes = N_classes;
    hydra->conv_frac_bit_shift =conv_frac_bit_shift;

    // Calculated attributes
    hydra->H       = hydra->G/2;
    hydra->len_feat_vec = hydra->H*hydra->K*hydra->N_dil*hydra->N_diff*hydra->N_feats*hydra->N_chan;

    // Allocate input vector
    hydra->inX      = (int16_t**) malloc(sizeof(int16_t*)*hydra->N_chan);
    hydra->inX_diff = (int16_t**) malloc(sizeof(int16_t*)*hydra->N_chan);

    for(int c=0; c < hydra->N_chan; c++) {
        hydra->inX[c]      = (int16_t*) malloc(sizeof(int16_t)*(hydra->lenX + 2*hydra->lenXpad+1));
        hydra->inX_diff[c] = (int16_t*) malloc(sizeof(int16_t)*(hydra->lenX + 2*hydra->lenXpad+1));

        // Initialize to zeros input vector
        for(int i=0; i <= hydra->lenX + 2*hydra->lenXpad; i++) {
            hydra->inX[c][i]      = (int16_t) (0);
            hydra->inX_diff[c][i] = (int16_t) (0);
        }
    }    

    // Allocate weight vector
    hydra->inW = (int16_t***) malloc(sizeof(int16_t**)*hydra->H);
    for(int h=0; h < hydra->H; h++) {
        hydra->inW[h] = (int16_t**) malloc(sizeof(int16_t*)*hydra->K);
        for(int k=0; k < hydra->K; k++) {
            hydra->inW[h][k] = (int16_t*) malloc(sizeof(int16_t)*(hydra->lenW));
        }
    }    

    // Allocate feature vector
    hydra->featVec = (int16_t*) malloc(sizeof(int16_t) * hydra->len_feat_vec); 

    // Allocate scaler attribute memory
    hydra->featMean = (int16_t*)  malloc(sizeof(int16_t) * hydra->len_feat_vec);
    hydra->featStd  = (uint8_t *) malloc(sizeof(uint8_t) * hydra->len_feat_vec);

    // Allocate classifier attribute structures
    hydra->classf_scores  = (int32_t*)  malloc(sizeof(int32_t)  * hydra->N_classes);
    hydra->classf_bias    = (int16_t*)  malloc(sizeof(int16_t)  * hydra->N_classes);
    hydra->classf_weights = (int16_t**) malloc(sizeof(int16_t*) * hydra->N_classes);

    for(int c=0; c < hydra->N_classes; c++) {
        hydra->classf_weights[c] = (int16_t*) malloc(sizeof(int16_t) * hydra->len_feat_vec);
    }

    return hydra;
}
