#include "../include/hydra.h"

void hydra_convolve(int16_t *inX, int8_t **inW, int16_t *featVec, uint16_t dil, Hydra* hydra, uint8_t curr_diff) {

    uint16_t   h,k,wi;
    uint16_t  xi;
    int32_t   conv_out = 0;
    int32_t   max, min;
    uint16_t  argmax=0, argmin=0;
    int16_t   *featVecPtr;
    int8_t    *inWptr[hydra->K];
    int16_t   *inXptr;

    for(h=0; h < hydra->H; h++) {
        featVecPtr = &(featVec[h*hydra->K*hydra->N_feats]);
        inXptr     = &inX[hydra->lenXpad-4*dil-4];

        for(k=0; k < hydra->K; k++) {
            inWptr[k] = &(inW[h][k*hydra->lenW]);
        }
        for(xi=0; xi < hydra->lenX - curr_diff; xi += 1) {
            // Reset the max and min
            max = INT32_MIN+1;
            min = INT32_MAX-1; 

            // Iterate over kernels in given group
            for(k=0; k < hydra->K; k++) {
                
                /* ALTERNATIVE 1 -- LOOP
                // Reset the convolutional output buffer, 
                //conv_out = 0;
                for(wi=0; wi < hydra->lenW; wi++) {
                    //conv_out += (int32_t)(inX[xi+hydra->lenXpad+(wi-4)*(dil+1)] * inWptr[k*hydra->lenW+wi]);
                    conv_out += (int32_t)(inXptr[xi+(wi)*(dil+1)] * inWptr[k][wi]);
                }
                */

                /* ALTERNATIVE 2 -- UNROLLED LOOP */
                conv_out = (int32_t)(inXptr[xi]             * inWptr[k][0] + 
                                     inXptr[xi +   (dil+1)] * inWptr[k][1] + 
                                     inXptr[xi + 2*(dil+1)] * inWptr[k][2] + 
                                     inXptr[xi + 3*(dil+1)] * inWptr[k][3] + 
                                     inXptr[xi + 4*(dil+1)] * inWptr[k][4] + 
                                     inXptr[xi + 5*(dil+1)] * inWptr[k][5] + 
                                     inXptr[xi + 6*(dil+1)] * inWptr[k][6] + 
                                     inXptr[xi + 7*(dil+1)] * inWptr[k][7] + 
                                     inXptr[xi + 8*(dil+1)] * inWptr[k][8]);

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
            featVecPtr[argmax*hydra->N_feats + 0] += (int16_t) (max >> hydra->conv_frac_bit_shift);
            featVecPtr[argmin*hydra->N_feats + 1] += 1;
        }
    }
}