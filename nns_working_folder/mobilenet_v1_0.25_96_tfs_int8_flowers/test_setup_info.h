#include "inputs/input_3.h"

#include "Source/genModel.c"

#define NN_OUTPUT_TYPE int8_t

#define NN_OUTPUT_TYPE_MIN -128

#define NN_OUTPUT_SIZE 5

#define INPUT_NUM 3

#define INPUT_SIZE_BYTE 27648

#define FOLDER_STR "mobilenet_v1_0.25_96_tfs_int8_flowers"
