#include <stdio.h>
#include <stdlib.h>
#include "../include/hydra_forward.h"
#include "../include/hydra_utils.h"
#include "../include/hydra_reset.h"


#define INPUT_LEN 140
#define WEIGH_LEN 9
#define NUM_CHAN  1
#define K (uint16_t) 8
#define G (uint16_t) 16
#define H G/2
#define MAX_DILATIONS 2
#define NUM_DIFFS     2
#define NUM_FEATS     2
#define LEN_FEATS     K*G*MAX_DILATIONS*NUM_DIFFS*NUM_FEATS

int main(void) {
    int16_t   **inX;
    int16_t   **inX_diff;
    int16_t  ***inW;
    int16_t    *featVec;
    uint16_t    pad_len;

    // Set the padding length
    pad_len = padding_len(WEIGH_LEN,MAX_DILATIONS);

    // Allocate input vector
    inX      = (int16_t**) malloc(sizeof(int16_t*)*NUM_CHAN);
    inX_diff = (int16_t**) malloc(sizeof(int16_t*)*NUM_CHAN);
    for(int c=0; c < NUM_CHAN; c++) {
        inX[c]      = (int16_t*) malloc(sizeof(int16_t)*(INPUT_LEN + 2*pad_len+1));
        inX_diff[c] = (int16_t*) malloc(sizeof(int16_t)*(INPUT_LEN + 2*pad_len+1));

        // Initialize to zeros input vector
        for(int i=0; i <= INPUT_LEN + 2*pad_len; i++) {
            inX[c][i]      = (int16_t) (0);
            inX_diff[c][i] = (int16_t) (0);
        }
    }    

    // Allocate weight vector
    inW = (int16_t***) malloc(sizeof(int16_t**)*H);
    for(int h=0; h < H; h++) {
        inW[h] = (int16_t**) malloc(sizeof(int16_t*)*K);
        for(int k=0; k < K; k++) {
            inW[h][k] = (int16_t*) malloc(sizeof(int16_t)*(WEIGH_LEN));
        }
    }    


    // Allocate feature vector
    featVec = (int16_t*) malloc(sizeof(int16_t) * LEN_FEATS); 

    hydra_reset(featVec, 
                INPUT_LEN, WEIGH_LEN, 
                H, K, 
                MAX_DILATIONS,  NUM_DIFFS, 
                NUM_CHAN, NUM_FEATS);

    // Read input vector
    FILE* fd = fopen("input.txt", "r");
    int ret = 0;

    for(int i=0; i < INPUT_LEN; i++) {
        ret = fscanf(fd, "%hd,", &inX[0][i+pad_len]);
        if(ret < 1) break;
    } 
    fclose(fd);

    // Read weights
    fd = fopen("weights.txt", "r");

    for(int h=0; h < H; h++) { 
        for(int k=0; k < K; k++) {
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


    // Generate Difference Vector

    // Calculate the differentiated version of inX. Parallelization point for OMP Parallel For. 
    // If single channel, there is no need for parallelization for small input vectors.
    for (int chan = 0; chan < NUM_CHAN; chan++) {
        for (int xi=0; xi < INPUT_LEN-1; xi++) {
            inX_diff[chan][xi+pad_len] = inX[chan][xi+1+pad_len]-inX[chan][xi+pad_len];
        }
    }


    hydra_forward(inX,
                  inX_diff, 
                  inW, 
                  featVec, 
                  INPUT_LEN, WEIGH_LEN, 
                  H, K, 
                  MAX_DILATIONS,  NUM_DIFFS, 
                  NUM_CHAN, NUM_FEATS);

    printf("Feats:\n");
    for(int i=0; i < 16; i++) {
        printf("%hd,", featVec[i]);
    }
    printf("\n");
}