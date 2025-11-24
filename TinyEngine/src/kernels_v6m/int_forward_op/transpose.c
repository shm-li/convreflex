#include <stdio.h>

#include "arm_nnfunctions.h"
#include "tinyengine_function.h"

// Do the same as TFLM: tensorflow/lite/kernels/internel/reference/transpose.h
tinyengine_status transpose_axis_1_2_int8(
		const int8_t *input, const uint16_t input_h, 
        const uint16_t input_w, const uint16_t input_c, 
        int8_t *output) {
	// for (int i = 0; i < size; ++i) {
	// 	printf("%d\n", input[i]);
	// 	fflush(stdout);
	// }
    const int8_t* input_ptr = input;
    int8_t* output_ptr = output;
	for (int i = 0; i < input_h; ++i) {
		for (int j = 0; j < input_w; ++j) {
            input_ptr = input + (i * input_w + j) * input_c;
            output_ptr = output + (j * input_h + i) * input_c;
			for (int k = 0; k < input_c; ++k) {
				output_ptr[k] = input_ptr[k];
			}
		}
	}
	// for (int i = 0; i < size; ++i) {
	// 	printf("%d\n", output[i]);
	// 	fflush(stdout);
	// }
}