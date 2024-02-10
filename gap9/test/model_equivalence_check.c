#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define _GNU_SOURCE
#include <fcntl.h>
#include <errno.h>
#include "../include/hydra.h"
#include "../include/hydra_defines.h"

int main(int argc, char *argv[]) {
    Hydra *hydra;
    int16_t *inXptr;
    int8_t  *inPtr_8;
    int16_t *inPtr_16;
    int ret;
    FILE* fd;

    // Initialize Hydra model
    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       NUM_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES, CONV_FRAC_BITS);


    // Read input vector
    int fdi = open(argv[1], O_RDONLY);
    int num_samples = atoi(argv[2]);
    printf("Feat Vect Len: %d. Processing %d samples\n", hydra->len_feat_vec, num_samples);

    if(fdi == -1)
        printf("Error opening input file");
    inXptr = (int16_t*) mmap(NULL, num_samples*(hydra->lenX+hydra->lenXpad*2+1)*2, PROT_READ, MAP_SHARED, fdi, 0);

    if(inXptr == MAP_FAILED)
        printf("MMAP Failed!\n");

    // Read weights
    fdi = open("./dist/weights.dat", O_RDONLY);
    inPtr_8 = (int8_t*) mmap(NULL, hydra->H*hydra->K*hydra->lenW, PROT_READ, MAP_SHARED, fdi, 0);
    for(int h=0; h < hydra->H; h++) { 
        memcpy(hydra->inW[h], &inPtr_8[h*hydra->K*hydra->lenW], hydra->lenW*hydra->K);
    }

    // Read scaler means 
    fdi = open("./dist/means.dat", O_RDONLY);
    inPtr_16 = (int16_t*) mmap(NULL, hydra->len_feat_vec, PROT_READ, MAP_SHARED, fdi, 0);
    memcpy(hydra->featMean, inPtr_16, 2*hydra->len_feat_vec);

    // Read scaler standard deviations
    fdi = open("./dist/stds.dat", O_RDONLY);
    inPtr_8 = (uint8_t*) mmap(NULL, hydra->len_feat_vec, PROT_READ, MAP_SHARED, fdi, 0);
    memcpy(hydra->featStd, inPtr_8, hydra->len_feat_vec);

    // Read classifier weights
    fdi = open("./dist/weights_classf.dat", O_RDONLY);
    inPtr_8 = (int8_t*) mmap(NULL, 2*hydra->N_classes*hydra->len_feat_vec, PROT_READ, MAP_SHARED, fdi, 0);
    for(int c=0; c < hydra->N_classes; c++) { 
        memcpy(hydra->classf_weights[c], &inPtr_8[c*hydra->len_feat_vec], hydra->len_feat_vec);
    }

    // Read classifier bias
    fdi = open("./dist/weights_bias.dat", O_RDONLY);
    inPtr_8 = (int8_t*) mmap(NULL, hydra->N_classes, PROT_READ, MAP_SHARED, fdi, 0);
    memcpy(hydra->classf_bias, inPtr_8, hydra->N_classes);


    // Dump features for validation
    int fdo = open("./dist/output.dat", O_RDWR | O_CREAT, (mode_t)0600);
    if(fdo == -1)
        printf("Error opening output file: %d", errno);
    ftruncate(fdo, num_samples*(hydra->N_classes)*4);

    if(ret < 0) {
        printf("Error allocating file!\n");
    }

    int32_t* outptr = (int32_t*) mmap(NULL, num_samples*(hydra->N_classes)*4, PROT_READ | PROT_WRITE, MAP_SHARED, fdo, 0); 

    if(outptr == MAP_FAILED)
        printf("MMAP Failed: %d\n", errno);

    // Process the current input vector.
    for(int i=0; i < num_samples; i++) {
        hydra_reset(hydra); 
        // Generate Difference Vector
        for (int chan = 0; chan < hydra->N_chan; chan++) {
            memcpy(&hydra->inX[0][hydra->lenXpad], &inXptr[i*hydra->lenX], hydra->lenX*2);

            for (int xi=0; xi < hydra->lenX-1; xi++) {
                hydra->inX_diff[chan][xi+hydra->lenXpad] = hydra->inX[chan][xi+1+hydra->lenXpad]-hydra->inX[chan][xi+hydra->lenXpad];
            }
        }

        hydra_forward(hydra);
        hydra_sparse_scale(hydra);
        hydra_classifier(hydra);

        // This is actually faster than memcpy
        outptr[i*hydra->N_classes + 0] = hydra->classf_scores[0];
        outptr[i*hydra->N_classes + 1] = hydra->classf_scores[1];
        outptr[i*hydra->N_classes + 2] = hydra->classf_scores[2];
        outptr[i*hydra->N_classes + 3] = hydra->classf_scores[3];
        outptr[i*hydra->N_classes + 4] = hydra->classf_scores[4];
    }

    msync(outptr, num_samples*(hydra->N_classes)*4, MS_SYNC);
    close(fdo);
    close(fdi);

}