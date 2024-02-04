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

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

// Problem Defines
#define INPUT_LEN 140
#define INPUT_SZ  2

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
#if defined(CONFIG_HELLOWORLD_CLUSTER)
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    struct pi_cluster_task cl_task;

    pi_cluster_conf_init(&cl_conf);
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        return -1;
    }
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, cluster_entry, NULL));
    pi_cluster_close(&cluster_dev);
#endif

    printf("Hello\n");


    // Setup perf counters
    uint32_t cycles, tim_cycles;
    pi_perf_conf(1 << PI_PERF_CYCLES || 1 << PI_PERF_ACTIVE_CYCLES);

    // Allocate space for buffer
    int16_t *values;
    int16_t sum=0;
    values = pi_l2_malloc(INPUT_LEN*INPUT_SZ);

    /************* SECTION 1: Setup of Readfs from File *************/
    static pi_fs_file_t *file[2] = {NULL};
    static struct pi_device fs;

    // Reads one line of input
    char flash_buffer[INPUT_SZ];

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

    /************* SECTION 2: Reading the input data into mem *************/
    // Open FD for Flash Section with input vector
    file[0] = pi_fs_open(&fs, STR(FILE_INPUT_DATA), 0);
    if (file[0] == NULL)
        return -2;
    
    // Checking that input file is correct
    for(int i = 0; i < INPUT_LEN; i++) {
        pi_fs_read(file[0], flash_buffer, 2);
        values[i] = (flash_buffer[1] << 8 | flash_buffer[0]);
        //printf("Read from file @[%d]: %d\n", i, values[i]);
    }

    // Close FD for Flash Section with input vector
    pi_fs_close(file[0]);
    pi_fs_unmount(&fs);
    /**********************************************************************/



    // Dummy task to test perf counters
    pi_perf_start();
    for(int i = 0; i < INPUT_LEN; i++) {
        sum += values[i]+1;
    }
    pi_perf_stop();


    // Output useful busy time
    cycles = pi_perf_read(PI_PERF_CYCLES);
    tim_cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
    printf("Perf: %d Cycles. Timer: %d cycles\n", cycles, tim_cycles);
    printf("Output value: %d\n", sum);

    
    /************* SECTION 3: Setup of Readfs from File *************/
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
    
    char *filename = "output_file.txt";
    file[1] = pi_fs_open(&host_fs, filename, PI_FS_FLAGS_WRITE);
    if (file[1] == NULL)
    {
        printf("Failed to open file, %s\n", STR(FILE1));
        return -4;
    }
    printf("Output file opened\n");
    /****************************************************************/


    /************* SECTION 4: Dumping output data to file *************/
    // Test output values by writing the input as it was
    for(int i = 0; i < INPUT_LEN; i++) {
        pi_fs_write(file[1], &values[i], sizeof(values[i]));
    }
    /******************************************************************/

    return 0;
}
