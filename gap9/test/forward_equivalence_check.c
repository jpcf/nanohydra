#include <stdio.h>
#include <stdlib.h>
#include "../include/hydra.h"


#define INPUT_LEN     140
#define WEIGH_LEN     9
#define NUM_CHAN      1
#define NUM_K         8
#define NUM_G         16
#define NUM_H         NUM_G/2
#define MAX_DILATIONS 4
#define NUM_DIFFS     2
#define NUM_FEATS     2
#define LEN_FEATS     NUM_K*NUM_H*MAX_DILATIONS*NUM_DIFFS*NUM_FEATS

int main(void) {
    int16_t   **inX;
    int16_t   **inX_diff;
    int16_t  ***inW;
    int16_t    *featVec;
    Hydra      *hydra;

    // Initialize the Hydra struct
    hydra = (Hydra*) malloc(sizeof(Hydra));
    hydra->lenX    = INPUT_LEN;
    hydra->lenW    = WEIGH_LEN;
    hydra->lenXpad = padding_len(WEIGH_LEN,MAX_DILATIONS);
    hydra->K       = NUM_K;
    hydra->G       = NUM_G;
    hydra->H       = NUM_G/2;
    hydra->N_dil   = MAX_DILATIONS;
    hydra->N_feats = NUM_FEATS;
    hydra->N_diff  = NUM_DIFFS;
    hydra->N_chan  = NUM_CHAN;
    hydra->len_feat_vec = hydra->H*hydra->K*hydra->N_dil*hydra->N_diff*hydra->N_feats*hydra->N_chan;

    // Allocate input vector
    inX      = (int16_t**) malloc(sizeof(int16_t*)*NUM_CHAN);
    inX_diff = (int16_t**) malloc(sizeof(int16_t*)*NUM_CHAN);
    for(int c=0; c < NUM_CHAN; c++) {
        inX[c]      = (int16_t*) malloc(sizeof(int16_t)*(INPUT_LEN + 2*hydra->lenXpad+1));
        inX_diff[c] = (int16_t*) malloc(sizeof(int16_t)*(INPUT_LEN + 2*hydra->lenXpad+1));

        // Initialize to zeros input vector
        for(int i=0; i <= INPUT_LEN + 2*hydra->lenXpad; i++) {
            inX[c][i]      = (int16_t) (0);
            inX_diff[c][i] = (int16_t) (0);
        }
    }    

    // Allocate weight vector
    inW = (int16_t***) malloc(sizeof(int16_t**)*hydra->H);
    for(int h=0; h < hydra->H; h++) {
        inW[h] = (int16_t**) malloc(sizeof(int16_t*)*hydra->K);
        for(int k=0; k < hydra->K; k++) {
            inW[h][k] = (int16_t*) malloc(sizeof(int16_t)*(WEIGH_LEN));
        }
    }    

    // Allocate feature vector
    featVec = (int16_t*) malloc(sizeof(int16_t) * LEN_FEATS); 

    hydra_reset(featVec, hydra); 

    // Read input vector
    FILE* fd = fopen("./dist/input.txt", "r");
    int ret = 0;

    for(int i=0; i < INPUT_LEN; i++) {
        ret = fscanf(fd, "%hd,", &inX[0][i+hydra->lenXpad]);
        if(ret < 1) break;
    } 
    fclose(fd);

    // Read weights
    fd = fopen("./dist/weights.txt", "r");

    for(int h=0; h < hydra->H; h++) { 
        for(int k=0; k < hydra->K; k++) {
            ret = fscanf(fd, "%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd\n", 
                &inW[h][k][0], 
                &inW[h][k][1], 
                &inW[h][k][2], 
                &inW[h][k][3], 
                &inW[h][k][4], 
                &inW[h][k][5], 
                &inW[h][k][6], 
                &inW[h][k][7],
                &inW[h][k][8]);
            if(ret < 9) break;
        }
    }
    fclose(fd);


    // Generate Difference Vector
    // Calculate the differentiated version of inX. Parallelization point for OMP Parallel For. 
    // If single channel, there is no need for parallelization for small input vectors.
    for (int chan = 0; chan < NUM_CHAN; chan++) {
        for (int xi=0; xi < INPUT_LEN-1; xi++) {
            inX_diff[chan][xi+hydra->lenXpad] = inX[chan][xi+1+hydra->lenXpad]-inX[chan][xi+hydra->lenXpad];
        }
    }

    hydra_forward(inX,
                  inX_diff, 
                  inW, 
                  featVec, 
                  hydra);

    // Dump features for validation
    fd = fopen("./dist/output.txt", "w");
    for(int i=0; i < LEN_FEATS; i++) {
        fprintf(fd, "%hd,", featVec[i]);
    }
    fclose(fd);
}