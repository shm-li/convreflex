#include <stdio.h>

#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status reduce_max_axis_1_2_int8(
		const int8_t *input, const uint16_t input_h, 
		const uint16_t input_w, const uint16_t input_c,
		// const uint16_t output_size,
		int8_t *output) {
	for (int i = 0; i < input_c; ++i) {
		output[i] = input[i];
	}
	for (int i = 0; i < input_h; ++i) {
		for (int j = 0; j < input_w; ++j) {
			for (int k = 0; k < input_c; ++k) {
				if (input[(i * input_w + j) * input_c + k] > output[k]) {
					output[k] = input[(i * input_w + j) * input_c + k];
				}
			}
		}
	}
}

tinyengine_status reduce_mean_axis_1_2_int8(
		const int8_t *input, const uint16_t input_h, 
		const uint16_t input_w, const uint16_t input_c,
		const int32_t multiplier, const int32_t shift,
		const int32_t input_zeropoint,
		const int32_t output_zeropoint,
		const int32_t min_val, const int32_t max_val,
		int8_t *output) {
	for (int k = 0; k < input_c; ++k) {
		int32_t sum = 0;
		for (int i = 0; i < input_h; ++i) {
			for (int j = 0; j < input_w; ++j) {
				// printf("\t%d %d: input[%d][%d][%d] = %d\n",(i * input_w + j), k, i, j, k, input[(i * input_w + j) * input_c + k]);
				// if (input[(i * input_w + j) * input_c + k] != -128) {
				// 	printf("\t\tnot -128\n");
				// }
				// fflush(stdout);
				sum += input[(i * input_w + j) * input_c + k];
			}
		}
		// int32_t res = (int8_t)(sum / (input_h * input_w));
		int32_t res = sum - input_zeropoint * input_h * input_w;
		res = \
			arm_nn_requantize(res, multiplier, shift) + output_zeropoint;
		res = (res < min_val) ? min_val : res;
		res = (res > max_val) ? max_val : res;
		output[k] = res;
	}
	// for (int k = 0; k < input_c; ++k) {
	// 	printf("\toutput[%d] = %d\n", k, output[k]);
	// }
}