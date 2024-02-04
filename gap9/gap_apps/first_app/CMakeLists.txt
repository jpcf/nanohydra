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
set(TARGET_SRCS first_app.c)

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
set(FILENAME_INPUT_DATA input_test.dat)
set(FILE_INPUT_DATA ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${FILENAME_INPUT_DATA})

# add compile definitions for the code to find the file
add_compile_definitions(FILE_INPUT_DATA=${FILENAME_INPUT_DATA})

# Add file to the readfs filesystem
readfs_add_files(FILES ${FILE_INPUT_DATA} FLASH mram PARTITION readfs_app)

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

## Include ssbl build here
include(ExternalProject)
ExternalProject_Add(ssbl
  SOURCE_DIR        "${CONFIG_GAP_SDK_HOME}/utils/ssbl"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/ssbl"
  INSTALL_COMMAND   ""
)

add_dependencies(${TARGET_NAME} ssbl)
