#include "../include/hydra.h"

void hydra_convolve(int16_t *inX, int16_t ***inW, int16_t *featVec, uint8_t dil, Hydra* hydra, uint8_t curr_diff) {

    uint8_t   h,k,wi;
    uint16_t  xi;
    int32_t   conv_out = 0;
    int16_t   max, min;
    uint16_t  argmax=0, argmin=0;

    for(h=0; h < hydra->H; h++) {
        for(xi=0; xi < hydra->lenX - curr_diff; xi += 1) {
            // Reset the max and min
            max = INT16_MIN;
            min = INT16_MAX; 

            // Iterate over kernels in given group
            for(k=0; k < hydra->K; k++) {
                // Reset the convolutional output buffer, 
                conv_out = 0;
                
                for(wi=0; wi < hydra->lenW; wi++) {
                    conv_out += (int32_t)(inX[xi+hydra->lenXpad+(wi-4)*(dil+1)] * inW[h][k][wi]);
                }

                // Determine if convolutional output is the new winning/losing kernel
                if(conv_out > max) {
                    // New winner kernel
                    max    = conv_out;
                    argmax = k;
                }
                if(conv_out < min) {
                    // New loser kernel
                    min    = conv_out;
                    argmin = k;
                }
            }

            // Hard count and soft count 
            featVec[h*hydra->K*hydra->N_feats + argmax*hydra->N_feats + 0] += max >> hydra->conv_frac_bit_shift;
            featVec[h*hydra->K*hydra->N_feats + argmin*hydra->N_feats + 1] += 1;
        }
    }
}