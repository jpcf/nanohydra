#include "../include/hydra.h"

#ifdef TARGET_GAP9
void hydra_convolve(void *args) {
    TeamForkArgs_T *kargs = (TeamForkArgs_T*) args;

    Hydra* hydra       = kargs->hydra;
    int16_t *inX       = kargs->inX; 
    int16_t  *inW      = kargs->inW; 
    int16_t *featVec   = kargs->featVec; 
    uint16_t dil       = kargs->dil;
    uint16_t curr_diff = kargs->diff_idx;

#else
void hydra_convolve(int16_t *inX, int8_t *inW, int16_t *featVec, uint16_t dil, Hydra* hydra, uint8_t curr_diff) {
#endif
    uint16_t   h,k,wi;
    uint16_t  xi;
    int32_t   conv_out[8] = {0};
    uint16_t  argmax=0, argmin=0;
    int16_t   featVecTmpMax[8];
    int16_t   featVecTmpMin[8];
    int16_t  *featVecPtr;
    int16_t   *inWptr[hydra->K];
    int16_t  *inXptr;

    uint8_t   shift = hydra->conv_frac_bit_shift;
    uint8_t   feats = hydra->N_feats;

    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    v2s opX_P1, opX_P2, opX_P3, opX_P4, opW_P1[8], opW_P2[8], opW_P3[8], opW_P4[8];
    int16_t opW_rem[8];

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
            opW_P1[k] = *((v2s*) &inW[h*hydra->K*hydra->lenW + k*hydra->lenW  ]);
            opW_P2[k] = *((v2s*) &inW[h*hydra->K*hydra->lenW + k*hydra->lenW+2]);
            opW_P3[k] = *((v2s*) &inW[h*hydra->K*hydra->lenW + k*hydra->lenW+4]);
            opW_P4[k] = *((v2s*) &inW[h*hydra->K*hydra->lenW + k*hydra->lenW+6]);
            opW_rem[k] = inW[h*hydra->K*hydra->lenW + k*hydra->lenW+8];
            featVecTmpMax[k] = 0;
            featVecTmpMin[k] = 0;
        }

        for(xi=0; xi < 140 - curr_diff; xi += 1) {
            
            // Reset the max and min
            argmin = 0;
            argmax = 0;

            opX_P1 = __builtin_pulp_pack2(inXptr[xi            ], inXptr[xi +   (dil+1)]);
            opX_P2 = __builtin_pulp_pack2(inXptr[xi + 2*(dil+1)], inXptr[xi + 3*(dil+1)]);
            opX_P3 = __builtin_pulp_pack2(inXptr[xi + 4*(dil+1)], inXptr[xi + 5*(dil+1)]);
            opX_P4 = __builtin_pulp_pack2(inXptr[xi + 6*(dil+1)], inXptr[xi + 7*(dil+1)]);

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
                conv_out[k] = 0;
                conv_out[k] = __builtin_pulp_sdotsp2(opX_P1 , opW_P1[k], conv_out[k]);
                conv_out[k] = __builtin_pulp_sdotsp2(opX_P2 , opW_P2[k], conv_out[k]);
                conv_out[k] = __builtin_pulp_sdotsp2(opX_P3 , opW_P3[k], conv_out[k]);
                conv_out[k] = __builtin_pulp_sdotsp2(opX_P4 , opW_P4[k], conv_out[k]);
                conv_out[k] += inXptr[xi + 8*(dil+1)] * opW_rem[k];
            }

            for(k=0; k < 8; k++) {
                if(conv_out[k] > conv_out[argmax]) {
                    // New winner kernel
                    argmax = k;
                }
                if(conv_out[k] < conv_out[argmin]) {
                    // New loser kernel
                    argmin = k;
                }
            }

            // Hard count and soft count. The accumulation is temporarily done here, as this avoids repeating
            // the access pointer arithmetic for lenX*H times. 
            featVecTmpMax[argmax] += (int16_t) (conv_out[argmax] >> shift);
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