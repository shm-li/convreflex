#include <stdio.h>

#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status requantize_int8_to_uint8(
		const char *inputptr, const uint16_t size,
		const int32_t effective_scale_multiplier, 
		const int32_t effective_scale_shift,
		const int32_t input_zeropoint, const int32_t output_zeropoint,
		const int32_t min_val, const int32_t max_val,
		uint8_t *output)
{
	int8_t* input = (int8_t*)inputptr;
	for (int i = 0; i < size; ++i) {
		int32_t input_num = input[i];
		int32_t num = (int32_t)input[i] - input_zeropoint;
		int32_t res =
			arm_nn_requantize(num, effective_scale_multiplier,
										effective_scale_shift) + 
			output_zeropoint;
		res = (res < min_val) ? min_val : res;
		res = (res > max_val) ? max_val : res;
		output[i] = (uint8_t)res;
		// printf("quantized %d to %d\n", input_num, output[i]);
		// fflush(stdout);
	}
}

tinyengine_status requantize_uint8_to_int8(
		const char *inputptr, const uint16_t size,
		const int32_t effective_scale_multiplier, 
		const int32_t effective_scale_shift,
		const int32_t input_zeropoint, const int32_t output_zeropoint,
		const int32_t min_val, const int32_t max_val,
		int8_t *output)
{
	uint8_t* input = (uint8_t*)inputptr;
	for (int i = 0; i < size; ++i) {
		const int32_t num = input[i] - input_zeropoint;
		int32_t res =
			arm_nn_requantize(num, effective_scale_multiplier,
										effective_scale_shift) + 
			output_zeropoint;
		res = (res < min_val) ? min_val : res;
		res = (res > max_val) ? max_val : res;
		output[i] = (uint8_t)res;
	}
}