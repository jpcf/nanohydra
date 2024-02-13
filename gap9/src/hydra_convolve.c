#include "../include/hydra.h"

#ifdef TARGET_GAP9
void hydra_convolve(void *args) {
    TeamForkArgs_T *kargs = (TeamForkArgs_T*) args;

    Hydra* hydra       = kargs->hydra;
    int16_t *inX       = kargs->inX; 
    int8_t *inW        = kargs->inW; 
    int16_t *featVec   = kargs->featVec; 
    uint16_t dil       = kargs->dil;
    uint16_t curr_diff = kargs->diff_idx;

#else
void hydra_convolve(int16_t *inX, int8_t *inW, int16_t *featVec, uint16_t dil, Hydra* hydra, uint8_t curr_diff) {
#endif
    uint16_t   h,k,wi;
    uint16_t  xi;
    int32_t   conv_out = 0;
    int32_t   max, min;
    uint16_t  argmax=0, argmin=0;
    int16_t   featVecTmpMax[8];
    int16_t   featVecTmpMin[8];
    int16_t   *featVecPtr;
    int8_t    *inWptr[hydra->K];
    int16_t   *inXptr;

    uint8_t   shift = hydra->conv_frac_bit_shift;
    uint8_t   feats = hydra->N_feats;

    // OMP for target x64
    //omp_set_num_threads(8);
    //#pragma omp parallel for private(h, k, xi, featVecPtr, featVecTmpMax, featVecTmpMin, inXptr, inWptr, min, max, conv_out) firstprivate(dil, curr_diff, argmin, argmax) shared(featVec, inX, inW, hydra)
    for(h=0; h < hydra->H; h++) {
        #ifdef TARGET_GAP9
        if(h == pi_core_id()) {
        #endif
        // Prefetch array at the right location, to avoid access pointer arithmetic for lenX*H times.
        featVecPtr = &(featVec[h*hydra->K*hydra->N_feats]);
        inXptr     = &inX[hydra->lenXpad-4*dil-4];

        //printf("featVectPtr addr=%p, core %d, chunk %d\n", featVec, pi_core_id(), h);
        for(k=0; k < hydra->K; k++) {
            inWptr[k] = &(inW[h*hydra->K*hydra->lenW + k*hydra->lenW]);
            featVecTmpMax[k] = 0;
            featVecTmpMin[k] = 0;
        }

        for(xi=0; xi < 140 - curr_diff; xi += 1) {
            
            // Reset the max and min
            max = INT32_MIN+1;
            min = INT32_MAX-1; 

            // Iterate over kernels in given group
            for(k=0; k < 8; k++) {
                
                /* ALTERNATIVE 1 -- LOOP
                // Reset the convolutional output buffer, 
                //conv_out = 0;
                for(wi=0; wi < hydra->lenW; wi++) {
                    //conv_out += (int32_t)(inX[xi+hydra->lenXpad+(wi-4)*(dil+1)] * inWptr[k*hydra->lenW+wi]);
                    conv_out += (int32_t)(inXptr[xi+(wi)*(dil+1)] * inWptr[k][wi]);
                }
                */

                /* ALTERNATIVE 2 -- UNROLLED LOOP */
                conv_out = (int32_t)(inXptr[xi           ] * inWptr[k][0] + 
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

            // Hard count and soft count. The accumulation is temporarily done here, as this avoids repeating
            // the access pointer arithmetic for lenX*H times. 
            featVecTmpMax[argmax] += (int16_t) (max >> shift);
            featVecTmpMin[argmin] += 1;
        }

        // The accumulation statistics for group h are saved in the main array.
        for(k=0; k < hydra->K; k++) {
            featVecPtr[k*feats + 0] += featVecTmpMax[k];
            featVecPtr[k*feats + 1] += featVecTmpMin[k];
        }
        #ifdef TARGET_GAP9
        }
        #endif
    }
}