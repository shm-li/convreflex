#include <stdint.h>
#include <stdio.h>
#include <string.h>  // For memcpy
#include "test_setup_info.h"

int main(int argc, char* argv[]) {
    printf("Starting %s, input %d\n", FOLDER_STR, INPUT_NUM);
    fflush(stdout);

    // Obtain a pointer to the model's input tensor
    char* input = getInput();

    char* output = getOutput();

    memcpy(input, input_data, INPUT_SIZE_BYTE);

    //printf("Starting\n");
    //fflush(stdout);
#ifdef RUN_BASELINE
    invoke_inf();
#else
    invoke_inf_omitting_redundancy();
#endif


    NN_OUTPUT_TYPE max_out = NN_OUTPUT_TYPE_MIN;
    uint8_t max_idx = 0;
    for (int i = 0; i < NN_OUTPUT_SIZE; ++i) {
        if (((NN_OUTPUT_TYPE *)(output))[i] > max_out) {
            max_out = ((NN_OUTPUT_TYPE *)(output))[i];
            max_idx = i;
        }
        printf("Output %d %f\n", ((NN_OUTPUT_TYPE *)(output))[i], ((NN_OUTPUT_TYPE *)(output))[i]);
        fflush(stdout);
    }
    printf("Label: %d, Max: %d\n", label, max_idx);
    fflush(stdout);
}
