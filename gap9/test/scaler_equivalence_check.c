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

int main(void) {
    Hydra *hydra;

    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       MAX_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES);
    hydra_reset(hydra); 

    printf("Feat Vect Len: %d\n", hydra->len_feat_vec);

    // Read input vector
    FILE* fd = fopen("./dist/input.txt", "r");
    int ret = 0;

    for(int i=0; i < hydra->len_feat_vec; i++) {
        ret = fscanf(fd, "%hd,", &(hydra->featVec[i]));
        if(ret < 1) break;
    } 
    fclose(fd);

    // Read means 
    fd = fopen("./dist/means.txt", "r");

    for(int i=0; i < hydra->len_feat_vec; i++) {
        ret = fscanf(fd, "%hd,", &(hydra->featMean[i]));
        if(ret < 1) break;
    } 
    fclose(fd);


    // Read standard deviations
    fd = fopen("./dist/stds.txt", "r");

    for(int i=0; i < hydra->len_feat_vec; i++) {
        ret = fscanf(fd, "%hhd,", &(hydra->featStd[i]));
        if(ret < 1) break;
    } 
    fclose(fd);

    // Process the current input vector.
    hydra_sparse_scale(hydra);

    // Dump features for validation
    fd = fopen("./dist/output.txt", "w");
    for(int i=0; i < hydra->len_feat_vec; i++) {
        fprintf(fd, "%hd,", hydra->featVec[i]);
    }
    fclose(fd);
}