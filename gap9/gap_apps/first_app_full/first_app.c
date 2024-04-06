/* 
 * Copyright (C) 2017 ETH Zurich, University of Bologna and GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Germain Haugou, ETH (germain.haugou@iis.ee.ethz.ch)
 */

#include "pmsis.h"
#include <bsp/bsp.h>
#include "../../include/hydra.h"
#include "../../include/hydra_defines.h"

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

// Problem Defines
#define INPUT_SZ  2
#define NUM_SAMPLES 500

#if defined(CONFIG_HELLOWORLD_CLUSTER)
void pe_entry(void *arg)
{
    printf("Hello from (%d, %d)\n", pi_cluster_id(), pi_core_id());
}

void cluster_entry(void *arg)
{
    pi_cl_team_fork(0, pe_entry, 0);
}
#endif

int main()
{

    // Setup perf counters
    float cycles_million = 0.0;
    uint32_t   cycles = 0, cycles_prev = 0;

    // Allocate space for buffer
    int16_t *values;
    int16_t sum=0;
    values = pi_l2_malloc(INPUT_LEN*INPUT_SZ);

    /************* SECTION 1: Setup of Readfs from File *************/
    static pi_fs_file_t *fd[2] = {NULL};
    static struct pi_device fs;

    // Reads one line of input
    char flash_buffer[32];

    struct pi_readfs_conf conf;
    pi_readfs_conf_init(&conf);

    // if using default layout, uncoment next line
    //conf.fs.partition_name = "readfs_mram";
    // if using custom layout, comment next line
    conf.fs.partition_name = "readfs_app";

    // Mounting the File System
    pi_open_from_conf(&fs, &conf);
    if (pi_fs_mount(&fs))
        return -1;
    printf("readfs mounted\n");
    /****************************************************************/


    printf("Data file path: '%s'!\n", STR(FILE_INPUT_DATA));
    printf("Weights file path: '%s'!\n", STR(FILE_INPUT_WEIGHTS));
    printf("Weights file path: '%s'!\n", STR(FILE_INPUT_CLASSF_W));
    printf("Weights file path: '%s'!\n", STR(FILE_INPUT_CLASSF_B));

    /************* SECTION 2: Init Hydra model, load weights *************/
    // Initialize Hydra model
    Hydra* hydra;
    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       NUM_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES, CONV_FRAC_BITS);

    printf("Hydra model successfully initialized!\n");

    // STEP A: Load RCK Weights
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_WEIGHTS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_WEIGHTS));
        return -2;
    }
    else {
        printf("Weights file opened successfully!\n");
    }

    for(int h=0; h < hydra->H; h++) {
        pi_fs_read(fd[0], hydra->inW[h], hydra->K*hydra->lenW);
    }
    pi_fs_close(fd[0]);

    // STEP B: Load Sparse Scaler Means
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_SS_MEANS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_SS_MEANS));
        return -2;
    }

    for(int f=0; f < hydra->len_feat_vec; f++) {
        pi_fs_read(fd[0], flash_buffer, 2);
        hydra->featMean[f] = (flash_buffer[1] << 8 | flash_buffer[0]);
        //printf("Read from file (featMean) @[%d]: %d\n", f, hydra->featMean[f]);
    }
    pi_fs_close(fd[0]);

    // STEP C: Load Sparse Scaler STDS
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_SS_STDS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_SS_STDS));
        return -2;
    }
    pi_fs_read(fd[0], hydra->featStd, hydra->len_feat_vec);

    for(int f=0; f < hydra->len_feat_vec; f++) {
        //printf("Read from file (featStd) @[%d]: %d\n", f, hydra->featStd[f]);
    }
    pi_fs_close(fd[0]);

    // STEP D: Load Classifier Weights
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_CLASSF_W), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_CLASSF_W));
        return -2;
    }

    for(int c=0; c < hydra->N_classes; c++) {
        for(int f=0; f < hydra->len_feat_vec; f++) {
            pi_fs_read(fd[0], &(hydra->classf_weights[c][f]), 1);
        }
    }
    pi_fs_close(fd[0]);

    // STEP E: Load Classifier Biases
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_CLASSF_B), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_CLASSF_B));
        return -2;
    }

    for(int c=0; c < hydra->N_classes; c++) {
        pi_fs_read(fd[0], &(hydra->classf_bias[c]), 1);
        //printf("Read from file (classfBias) @[%d]: %d\n", c, hydra->classf_bias[c]);
    }
    pi_fs_close(fd[0]);

    /************* SECTION 3a: Opening Input Data file descriptor *************/
    // Open FD for Flash Section with input vector
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_DATA), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_DATA));
        return -2;
    }
    /**********************************************************************/
    

    /************* SECTION 3b: Setup of Host FS (for output dump) *************/
    /** HostFs dump to PC **/
    pi_device_t host_fs;
    struct pi_hostfs_conf hostfs_conf;
    pi_hostfs_conf_init(&hostfs_conf);

    pi_open_from_conf(&host_fs, &hostfs_conf);

    if (pi_fs_mount(&host_fs))
    {
        printf("Failed to mount host fs\n");
        return -3;
    }
    printf("Hostfs mounted\n");
    
    char *filename = "output.dat";
    fd[1] = pi_fs_open(&host_fs, filename, PI_FS_FLAGS_WRITE);
    if (fd[1] == NULL)
    {
        printf("Failed to open file, OUTPUT\n");
        return -4;
    }
    printf("Output file opened\n");
    /**************************************************************************/


    /************* SECTION 4: Performing forward passes on test samples *************/
    for(int s=0; s < NUM_SAMPLES; s++) {
        /************* SECTION 4a: Reading the input data into mem *************/
        for(int i = 0; i < INPUT_LEN; i++) {
            pi_fs_read(fd[0], flash_buffer, 2);
            hydra->inX[0][i+hydra->lenXpad] = (flash_buffer[1] << 8 | flash_buffer[0]);
            //printf("Read from file @[%d]: %d\n", i, hydra->inX[0][i]);
        }
        for (int xi=0; xi < hydra->lenX-1; xi++) {
            hydra->inX_diff[0][xi+hydra->lenXpad] = hydra->inX[0][xi+1+hydra->lenXpad]-hydra->inX[0][xi+hydra->lenXpad];
        }
        /***********************************************************************/


        /************* SECTION 4b: Performing forward pass *************/
        hydra_reset(hydra);
        pi_perf_conf(1 << PI_PERF_CYCLES || 1 << PI_PERF_ACTIVE_CYCLES);
        pi_perf_start();
        hydra_forward(hydra);
        hydra_sparse_scale(hydra);
        hydra_classifier(hydra);
        pi_perf_stop();
        /***************************************************************/

        /************* SECTION 4c: Collect benchmarks *************/
        cycles = pi_perf_read(PI_PERF_CYCLES);
        cycles_million  += (float)(cycles-cycles_prev) / 1000000;
        if(s % 20 == 0)
            printf("Processed %6d samples. # Cycles: %ld\n", s, cycles-cycles_prev);
        cycles_prev = cycles;
        /**********************************************************/

        /************* SECTION 4d: Dumping output data to file *************/
        // Test output values by writing the input as it was
        for(int i = 0; i < hydra->N_classes; i++) {
            flash_buffer[0] = (hydra->classf_scores[i]      ) & 0xFF;
            flash_buffer[1] = (hydra->classf_scores[i] >>  8) & 0xFF;
            flash_buffer[2] = (hydra->classf_scores[i] >> 16) & 0xFF;
            flash_buffer[3] = (hydra->classf_scores[i] >> 24) & 0xFF;
            pi_fs_write(fd[1], &hydra->classf_scores[i], sizeof(hydra->classf_scores[i]));
        }
        /******************************************************************/

    }

    // Close FD for Flash Section with input vector
    pi_fs_close(fd[1]);
    pi_fs_unmount(&fs);

    float avg_cycles = (float)cycles_million / NUM_SAMPLES;
    float avg_inf_time_ms = (avg_cycles  * 1e6) / 320e6 * 1e3;

    printf("Average inference time: %.3f ms. -- In raw perf cycles: ~= %.3f Million cycles\n", avg_inf_time_ms, avg_cycles);

    return 0;
}
