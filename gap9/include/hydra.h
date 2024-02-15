#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#ifdef TARGET_GAP9
#include "pmsis.h"
#include <bsp/bsp.h>
#endif

#include "hydra_defines.h"

typedef struct Hydra {
    // Memory allocations
    int16_t   **inX;
    int16_t   **inX_diff;
    int8_t     *inW;
    int16_t    *featVec;
    int8_t   **classf_weights;
    int8_t    *classf_bias;
    int32_t    *classf_scores;

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
    uint8_t  conv_frac_bit_shift;

    // Classifier Attributes
    uint8_t N_classes;

    // Scaler Attributes
    int16_t *featMean;
    uint8_t *featStd;

} Hydra;

#ifdef TARGET_GAP9
typedef struct {
    int16_t * __restrict__ inX;
    int8_t  * __restrict__ inW;
    int16_t * __restrict__ featVec;
    uint8_t  dil;
    Hydra*   hydra;
    uint8_t  diff_idx;
} TeamForkArgs_T;
#else
#endif


Hydra* hydra_init(
    uint16_t  lenX,
    uint16_t  lenW,
    uint16_t  H,     
    uint16_t  G,
    uint8_t   N_dil,
    uint8_t   N_diff,
    uint8_t   N_chan, 
    uint8_t   N_feats,
    uint8_t   N_classes,
    uint8_t   conv_frac_bit_shift);

void hydra_reset(Hydra *hydra);

#ifdef TARGET_GAP9
void hydra_convolve(void* args);
#else
void hydra_convolve(int16_t   *inX, 
                    int8_t    *inW, 
                    int16_t   *featVec, 
                    uint16_t   dil,
                    Hydra     *hydra,
                    uint8_t    diff_idx
                    );
#endif

void hydra_forward(Hydra *hydra);

void hydra_sparse_scale(Hydra *hydra); 

void hydra_classifier(Hydra* hydra);

uint16_t padding_len(uint16_t lenW, uint16_t dilation);

uint16_t generate_dilation_val(uint16_t dil_idx);