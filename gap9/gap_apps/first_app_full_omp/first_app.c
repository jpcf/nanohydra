
#include "pmsis.h"
#include <bsp/bsp.h>
#include "../../include/hydra.h"
#include "../../include/hydra_defines.h"

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

// Problem Defines
#define INPUT_SZ  2
#define NUM_SAMPLES 50
#define BUFFER_SZ   1200

#ifdef PARALLELIZE
static PI_L1 RCKINT  inX[BUFFER_SZ], inX_diff[BUFFER_SZ], inW[BUFFER_SZ];
static PI_L1 int16_t featVec[2*BUFFER_SZ];

void hydra_forward_gap9(void *args) {
    
    uint8_t dil_idx;
    uint8_t diff_idx;
    uint8_t chan;
    uint16_t dil;

    Hydra* hydra = (Hydra *) args;
    
    // Copy Input Vector and Weights into L1
    pi_cl_dma_copy_t copy_L2_to_L1_inX, copy_L2_to_L1_inX_diff, copy_L2_to_L1_inW;
    copy_L2_to_L1_inX.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_inX.merge = 0;
    copy_L2_to_L1_inX.size  = (uint16_t) hydra->lenX*sizeof(RCKINT);
    copy_L2_to_L1_inX.id    = 0;
    copy_L2_to_L1_inX.ext   = (uint32_t) &(hydra->inX[0][hydra->lenXpad]);
    copy_L2_to_L1_inX.loc   = (uint32_t) &(inX[hydra->lenXpad]);

    copy_L2_to_L1_inX_diff.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_inX_diff.merge = 0;
    copy_L2_to_L1_inX_diff.size  = (uint16_t) (hydra->lenX-1)*sizeof(RCKINT);
    copy_L2_to_L1_inX_diff.id    = 1;
    copy_L2_to_L1_inX_diff.ext   = (uint32_t) &(hydra->inX_diff[0][hydra->lenXpad]);
    copy_L2_to_L1_inX_diff.loc   = (uint32_t) &(inX_diff[hydra->lenXpad]);

    copy_L2_to_L1_inW.dir   = PI_CL_DMA_DIR_EXT2LOC;
    copy_L2_to_L1_inW.merge = 0;
    copy_L2_to_L1_inW.size  = (uint16_t) 2*hydra->lenW*hydra->K*hydra->H;
    copy_L2_to_L1_inW.id    = 2;
    copy_L2_to_L1_inW.ext   = (uint32_t) hydra->inW;
    copy_L2_to_L1_inW.loc   = (uint32_t) inW;

    pi_cl_dma_memcpy(&copy_L2_to_L1_inX);
    pi_cl_dma_memcpy(&copy_L2_to_L1_inX_diff);
    pi_cl_dma_memcpy(&copy_L2_to_L1_inW);
    pi_cl_dma_wait(&copy_L2_to_L1_inX);
    pi_cl_dma_wait(&copy_L2_to_L1_inX_diff);
    pi_cl_dma_wait(&copy_L2_to_L1_inW);

    for(int i=0; i < hydra->len_feat_vec; i++) {
        featVec[i] = 0;
    }

    TeamForkArgs_T fork_args;
    fork_args.inW   = inW;
    fork_args.hydra = hydra;

    for (dil_idx = 0; dil_idx < hydra->N_dil; dil_idx++) {
        dil = generate_dilation_val(dil_idx);
        for (diff_idx = 0; diff_idx < hydra->N_diff; diff_idx++) {
            for (chan = 0; chan < hydra->N_chan; chan++) {
                fork_args.dil = dil;
                fork_args.diff_idx = diff_idx;
                fork_args.featVec  = &(featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]);
                if(diff_idx == 0) {
                    fork_args.inX = inX;
                    pi_cl_team_fork(pi_cl_cluster_nb_cores(), hydra_convolve, &fork_args);
                }
                else {
                    fork_args.inX = inX_diff;
                    pi_cl_team_fork(pi_cl_cluster_nb_cores(), hydra_convolve, &fork_args);
                }
            }
        }
    }

    // Copy out feature vector from L1 into L2
    pi_cl_dma_copy_t copy_L1_to_L2;
    copy_L1_to_L2.dir   = PI_CL_DMA_DIR_LOC2EXT;
    copy_L1_to_L2.merge = 0;
    copy_L1_to_L2.size  = (uint16_t) 2*hydra->len_feat_vec;
    copy_L1_to_L2.id    = 0;
    copy_L1_to_L2.ext   = (uint32_t) hydra->featVec;
    copy_L1_to_L2.loc   = (uint32_t) featVec;

    pi_cl_dma_memcpy(&copy_L1_to_L2);
    pi_cl_dma_wait(&copy_L1_to_L2);
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

    /************* SECTION 4: Setup Cluster Task*************/
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;  
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |   
                       // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                       PI_CLUSTER_ICACHE_PREFETCH_ENABLE |      
                       // Enable the icache for all the cores
                       PI_CLUSTER_ICACHE_ENABLE;
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    /* Prepare cluster task and send it to cluster. */
    struct pi_cluster_task cl_task;
    /************* SECTION 2: Init Hydra model, load weights *************/
    // Initialize Hydra model
    Hydra *hydra;
    hydra = hydra_init(INPUT_LEN, WEIGH_LEN, NUM_K, NUM_G,
                       NUM_DILATIONS, NUM_DIFFS, NUM_CHAN,
                       NUM_FEATS, NUM_CLASSES, CONV_FRAC_BITS);


    #ifdef PARALLELIZE
    // Zero padding to input vectors in L1
    for(int i=0; i < hydra->lenXpad; i++) {
        inX[i] = 0;
    }
    for(int i=hydra->lenXpad+hydra->lenX; i < 2*hydra->lenXpad+hydra->lenX+1; i++) {
        inX[i] = 0;
    }
    for(int i=0; i < hydra->lenXpad; i++) {
        inX_diff[i] = 0;
    }
    for(int i=hydra->lenXpad+hydra->lenX-1; i < 2*hydra->lenXpad+hydra->lenX; i++) {
        inX_diff[i] = 0;
    }
    printf("Hydra model successfully initialized!\n");
    pi_cluster_close(&cluster_dev);
    #endif
    
    // STEP A: Load RCK Weights
    fd[0] = pi_fs_open(&fs, STR(FILE_INPUT_WEIGHTS), 0);
    if (fd[0] == NULL) {
        printf("Error opening file '%s'!\n", STR(FILE_INPUT_WEIGHTS));
        return -2;
    }
    else {
        printf("Weights file opened successfully!\n");
    }

    pi_fs_read(fd[0], hydra->inW, 2*hydra->H*hydra->K*hydra->lenW);
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


    /************* SECTION 5: Performing forward passes on test samples *************/
    for(int s=0; s < NUM_SAMPLES; s++) {
        /************* SECTION 4a: Reading the input data into mem *************/
        for(int i = 0; i < INPUT_LEN; i++) {
            if(sizeof(RCKINT) == 2) {
                pi_fs_read(fd[0], flash_buffer, 2);
                hydra->inX[0][i+hydra->lenXpad] = (flash_buffer[1] << 8 | flash_buffer[0]);
            }
            else {
                pi_fs_read(fd[0], flash_buffer, 1);
                hydra->inX[0][i+hydra->lenXpad] = flash_buffer[0];
            }
            //printf("Read from file @[%d]: %d\n", i, hydra->inX[0][i]);
        }
        for (int xi=0; xi < hydra->lenX-1; xi++) {
            hydra->inX_diff[0][xi+hydra->lenXpad] = hydra->inX[0][xi+1+hydra->lenXpad]-hydra->inX[0][xi+hydra->lenXpad];
        }
        /***********************************************************************/
        
        pi_open_from_conf(&cluster_dev, &cl_conf);
        if (pi_cluster_open(&cluster_dev))
        {
            printf("Cluster open failed !\n");
            pmsis_exit(-1);
        }

        hydra_reset(hydra);
        pi_perf_start();
        #ifdef PARALLELIZE
        pi_cluster_task(&cl_task, hydra_forward_gap9, hydra);
        pi_cluster_send_task_to_cl(&cluster_dev, &cl_task);
        pi_cluster_close(&cluster_dev);
        #else
        hydra_forward(hydra);
        #endif
        hydra_sparse_scale(hydra);
        hydra_classifier(hydra);
        pi_perf_stop();

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
    float avg_inf_time_ms = (avg_cycles  * 1e6) / 100e6 * 1e3;

    printf("Average inference time: %.3f ms. -- In raw perf cycles: ~= %.3f Million cycles\n", avg_inf_time_ms, avg_cycles);

    return 0;
}
