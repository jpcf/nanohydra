#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define QN_SCALER =  3 
#define QM_SCALER = 12

typedef struct Hydra {
    // Memory allocations
    int16_t   **inX;
    int16_t   **inX_diff;
    int16_t  ***inW;
    int16_t    *featVec;
    int16_t   **classf_weights;
    int16_t    *classf_bias;
    int16_t    *classf_scores;

    // Attributes
    uint16_t lenX;  
    uint16_t lenW;
    uint16_t lenXpad;
    uint16_t H;     
    uint16_t K;
    uint16_t G;
    uint8_t  N_dil; 
    uint8_t  N_diff; 
    uint8_t  N_chan; 
    uint8_t  N_feats;
    uint16_t len_feat_vec;

    // Classifier Attributes
    uint8_t N_classes;

} Hydra;

Hydra* hydra_init(
    uint16_t  lenX,
    uint16_t  lenW,
    uint16_t  H,     
    uint16_t  G,
    uint8_t   N_dil,
    uint8_t   N_diff,
    uint8_t   N_chan, 
    uint8_t   N_feats,
    uint8_t   N_classes);

void hydra_reset(Hydra *hydra);

void hydra_convolve(int16_t   *inX, 
                    int16_t ***inW, 
                    int16_t   *featVec, 
                    uint8_t    dil,
                    Hydra     *hydra,
                    uint8_t    diff_idx
                    );

void hydra_forward(Hydra *hydra);

void hydra_sparse_scale(int16_t featVec, float featMean, float featStd, uint16_t lenFeatVec); 

void hydra_classifier(Hydra* hydra);

uint16_t padding_len(uint16_t lenW, uint16_t dilation);

uint16_t generate_dilation_val(uint16_t dil_idx);