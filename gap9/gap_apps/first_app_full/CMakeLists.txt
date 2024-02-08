# Copyright (c) 2022 GreenWaves Technologies SAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of GreenWaves Technologies SAS nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.19)

###############################################################################
# Panel Control
###############################################################################
set(TARGET_NAME "first_app")
set(HYDRA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(TARGET_SRCS first_app.c  ${HYDRA_SRCS}/hydra_init.c 
                             ${HYDRA_SRCS}/hydra_reset.c 
                             ${HYDRA_SRCS}/hydra_utils.c 
                             ${HYDRA_SRCS}/hydra_forward.c 
                             ${HYDRA_SRCS}/hydra_convolve.c 
                             ${HYDRA_SRCS}/hydra_sparse_scale.c 
                             ${HYDRA_SRCS}/hydra_classifier.c)

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

project(${TARGET_NAME} C ASM)
add_executable(${TARGET_NAME} ${TARGET_SRCS})

###############################################################################
# App's options interpretation
###############################################################################
if (CONFIG_HELLOWORLD_CLUSTER)
    message(STATUS "[${TARGET_NAME} Options] Cluster enabled")
    target_compile_options(${TARGET_NAME} PRIVATE "-DCONFIG_HELLOWORLD_CLUSTER=1")
else()
    message(STATUS "[${TARGET_NAME} Options] Cluster disabled")
endif()

###############################################################################
# Adding a readfs file --> defaut layout | comment out for version with layout
###############################################################################

# get proper file path
set(FILENAME_INPUT_DATA    input_train.dat)
set(FILENAME_INPUT_WEIGHTS weights.dat)
set(FILENAME_INPUT_SS_MEANS    means.dat)
set(FILENAME_INPUT_SS_STDS    stds.dat)
set(FILENAME_INPUT_CLASSF_B    weights_bias.dat)
set(FILENAME_INPUT_CLASSF_W    weights_classf.dat)
set(FILE_INPUT_DATA     ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_DATA})
set(FILE_INPUT_WEIGHTS  ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_WEIGHTS})
set(FILE_INPUT_SS_MEANS ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_SS_MEANS})
set(FILE_INPUT_SS_STDS  ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_SS_STDS})
set(FILE_INPUT_CLASSF_B ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_CLASSF_B})
set(FILE_INPUT_CLASSF_W ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_CLASSF_W})

# add compile definitions for the code to find the file
add_compile_definitions(FILE_INPUT_DATA=${FILENAME_INPUT_DATA})
add_compile_definitions(FILE_INPUT_WEIGHTS=${FILENAME_INPUT_WEIGHTS})
add_compile_definitions(FILE_INPUT_SS_MEANS=${FILENAME_INPUT_SS_MEANS})
add_compile_definitions(FILE_INPUT_SS_STDS=${FILENAME_INPUT_SS_STDS})
add_compile_definitions(FILE_INPUT_CLASSF_B=${FILENAME_INPUT_CLASSF_B})
add_compile_definitions(FILE_INPUT_CLASSF_W=${FILENAME_INPUT_CLASSF_W})

# Pre-compiler switches to enable GAP9-port specific funcionalities
add_compile_definitions(TARGET_GAP9=1)

# Add file to the readfs filesystem
readfs_add_files(FILES ${FILE_INPUT_DATA} ${FILE_INPUT_WEIGHTS} ${FILE_INPUT_SS_MEANS} ${FILE_INPUT_SS_STDS} ${FILE_INPUT_CLASSF_W} ${FILE_INPUT_CLASSF_B} FLASH mram PARTITION readfs_app)

###############################################################################
# Setup the SSBL
###############################################################################

list(APPEND GAPY_RUNNER_ARGS
    --flash-property=${CMAKE_BINARY_DIR}/ssbl/ssbl@mram:ssbl:binary
    --flash-property=${CMAKE_BINARY_DIR}/first_app@mram:app:binary)
###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
set_source_files_properties(${TARGET_SRCS} PROPERTIES COMPILE_OPTIONS "-O3")

## Include ssbl build here
include(ExternalProject)
ExternalProject_Add(ssbl
  SOURCE_DIR        "${CONFIG_GAP_SDK_HOME}/utils/ssbl"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/ssbl"
  INSTALL_COMMAND   ""
)

add_dependencies(${TARGET_NAME} ssbl)
