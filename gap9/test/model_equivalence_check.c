#include <stdio.h>
#include <stdlib.h>
#include "../include/hydra.h"


#define INPUT_LEN     140
#define WEIGH_LEN     9
#define NUM_CHAN      1
#define NUM_K         8
#define NUM_G         16
#define MAX_DILATIONS 5
#define NUM_DIFFS     2
#define NUM_FEATS     2
#define NUM_CLASSES   5
#define CONV_FRAC_BITS 10

int main(void) {
    Hydra *hydra;

    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       MAX_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES, CONV_FRAC_BITS);
    hydra_reset(hydra); 

    printf("Feat Vect Len: %d\n", hydra->len_feat_vec);

    // Read input vector
    FILE* fd = fopen("./dist/input.txt", "r");
    int ret = 0;

    for(int i=0; i < hydra->lenX; i++) {
        ret = fscanf(fd, "%hd,", &(hydra->inX[0][i+hydra->lenXpad]));
        if(ret < 1) break;
    } 
    fclose(fd);

    // Read weights
    fd = fopen("./dist/weights.txt", "r");
    for(int h=0; h < hydra->H; h++) { 
        for(int k=0; k < hydra->K; k++) {
            ret = fscanf(fd, "%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd\n", 
                &(hydra->inW[h][k][0]), 
                &(hydra->inW[h][k][1]), 
                &(hydra->inW[h][k][2]), 
                &(hydra->inW[h][k][3]), 
                &(hydra->inW[h][k][4]), 
                &(hydra->inW[h][k][5]), 
                &(hydra->inW[h][k][6]), 
                &(hydra->inW[h][k][7]),
                &(hydra->inW[h][k][8]));
            if(ret < 9) break;
        }
    }
    fclose(fd);

    // Read scaler means 
    fd = fopen("./dist/means.txt", "r");

    for(int i=0; i < hydra->len_feat_vec; i++) {
        ret = fscanf(fd, "%hd,", &(hydra->featMean[i]));
        if(ret < 1) break;
    } 
    fclose(fd);

    // Read scaler standard deviations
    fd = fopen("./dist/stds.txt", "r");
    for(int i=0; i < hydra->len_feat_vec; i++) {
        ret = fscanf(fd, "%hhd,", &(hydra->featStd[i]));
        if(ret < 1) break;
    } 
    fclose(fd);

    // Read classifier weights
    fd = fopen("./dist/weights_classf.txt", "r");

    for(int c=0; c < hydra->N_classes; c++) { 
        for(int f=0; f < hydra->len_feat_vec; f++) {
            ret = fscanf(fd, "%hd,", &(hydra->classf_weights[c][f]));
            if(ret < 1) break;
        }
    }
    fclose(fd);

    // Read classifier bias
    fd = fopen("./dist/bias_classf.txt", "r");

    for(int c=0; c < hydra->N_classes; c++) { 
        ret = fscanf(fd, "%hd,", &(hydra->classf_bias[c]));
        if(ret < 1) break;
    }
    fclose(fd);

    // Generate Difference Vector
    for (int chan = 0; chan < hydra->N_chan; chan++) {
        for (int xi=0; xi < hydra->lenX-1; xi++) {
            hydra->inX_diff[chan][xi+hydra->lenXpad] = hydra->inX[chan][xi+1+hydra->lenXpad]-hydra->inX[chan][xi+hydra->lenXpad];
        }
    }

    // Process the current input vector.
    hydra_forward(hydra);
    hydra_sparse_scale(hydra);
    hydra_classifier(hydra);

    // Dump features for validation
    fd = fopen("./dist/output.txt", "w");
    for(int i=0; i < hydra->N_classes; i++) {
        fprintf(fd, "%d,", hydra->classf_scores[i]);
    }
    fclose(fd);
}