#include <stdint.h>

void hydra_convolve(int16_t *inX, int16_t ***inW, int16_t *featVec, uint8_t dil, uint16_t lenX, uint8_t lenW, uint8_t lenXpad, uint8_t H, uint8_t K, uint8_t F) {

    uint8_t   h,k,wi;
    uint16_t  xi;
    int32_t   conv_out = 0;
    int16_t   max, min;
    uint16_t  argmax, argmin;

    for(h=0; h < H; h++) {
        for(xi=0; xi < lenX; xi += 1) {
            // Reset the max and min
            max = INT16_MIN;
            min = INT16_MAX; 

            // Iterate over kernels in given group
            for(k=0; k < K; k++) {
                // Reset the convolutional output buffer, 
                conv_out = 0;
                
                // Convolve for the current X point. Doing it this way prevents the need to keep the full convolutional
                // output in memory, since each convolutional output is independent of the others. This is a possible point
                // for using SIMD.
                for(wi=0; wi < lenW; wi++) {
                    conv_out += (int32_t)(inX[xi+lenXpad+(wi-4)*(dil+1)] * inW[h][k][wi]);
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

            // Hard count and **NOT USED** soft count 
            featVec[h*K*F + argmax*F + 0] += max;
            featVec[h*K*F + argmin*F + 1] += 1;
        }
    }
}