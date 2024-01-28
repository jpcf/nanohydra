#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
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

int main(int argc, char *argv[]) {
    Hydra *hydra;
    int16_t *inXptr;

    // Initialize Hydra model
    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       MAX_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES, CONV_FRAC_BITS);


    // Read input vector
    int fdi = open(argv[1], O_RDONLY);
    int num_samples = atoi(argv[2]);
    printf("Feat Vect Len: %d. Processing %d samples\n", hydra->len_feat_vec, num_samples);

    if(fdi == -1)
        printf("Error opening input file");
    inXptr = (int16_t*) mmap(NULL, num_samples*(INPUT_LEN+hydra->lenXpad*2+1)*2, PROT_READ, MAP_SHARED, fdi, 0)
    ;

    if(inXptr == MAP_FAILED)
        printf("MMAP Failed!\n");

    // Read weights
    FILE* fd = fopen("./dist/weights.txt", "r");
    int ret;
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

    // Dump features for validation
    fd = fopen("./dist/output.txt", "w");

    // Process the current input vector.
    for(int i=0; i < num_samples; i++) {

        hydra_reset(hydra); 

        // Generate Difference Vector
        for (int chan = 0; chan < hydra->N_chan; chan++) {
            for (int xi=0; xi < hydra->lenX; xi++) {
                hydra->inX[0][hydra->lenXpad+xi] = inXptr[i*INPUT_LEN+xi];
            }
            for (int xi=0; xi < hydra->lenX-1; xi++) {
                hydra->inX_diff[chan][xi+hydra->lenXpad] = hydra->inX[chan][xi+1+hydra->lenXpad]-hydra->inX[chan][xi+hydra->lenXpad];
            }
        }

        /*
        // Useful for debugging, if needed
        printf("Just Checking [%d] = %d %d %d ... %d %d %d\n", i, 
                hydra->inX[0][hydra->lenXpad+0], hydra->inX[0][hydra->lenXpad+1], hydra->inX[0][hydra->lenXpad+2],
                hydra->inX[0][hydra->lenXpad+137], hydra->inX[0][hydra->lenXpad+138], hydra->inX[0][hydra->lenXpad+139]
                );
        */
        hydra_forward(hydra);
        hydra_sparse_scale(hydra);
        hydra_classifier(hydra);

        for(int i=0; i < hydra->N_classes; i++) {
            fprintf(fd, "%d,", hydra->classf_scores[i]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}