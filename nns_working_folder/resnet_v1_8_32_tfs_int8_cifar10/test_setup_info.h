#include "inputs/input_3.h"

#include "Source/genModel.c"

#define NN_OUTPUT_TYPE int8_t

#define NN_OUTPUT_TYPE_MIN -128

#define NN_OUTPUT_SIZE 10

#define INPUT_NUM 3

#define INPUT_SIZE_BYTE 3072

#define FOLDER_STR "resnet_v1_8_32_tfs_int8_cifar10"
