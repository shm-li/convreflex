#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status convolve_1x1_s8_clamping_pred(const int8_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const int8_t *kernel,
		const uint8_t *termination_steps,
		const int32_t *termination_bounds,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, int8_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, int16_t *runtime_buf) {


	(void) input_x;
	(void) input_y;

	int8_t *out = output;
	const int32_t num_elements = output_x * output_y;

	for (int32_t i_element = 0; i_element < num_elements; i_element++) {
		// int32_t i_ch_out;
		// const int8_t *ker_ptr = kernel;
		int8_t *src = &input[i_element * input_ch];
		int16_t *dst = runtime_buf;

		// for (int32_t i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
		// 	int32_t sum = bias[i_ch_out];
		// 	uint16_t ker_ch_count = input_ch * DIM_KER_X * DIM_KER_Y;
		// 	const int8_t *in_ptr = &input[i_element * input_ch];

		// 	while (ker_ch_count) {
		// 		int16_t in = *in_ptr++;
		// 		int8_t ker = *ker_ptr++;
		// 		sum += (in + input_offset) * ker;
		// 		// printf("\t sum += %d * (%d + %d), now %d. ker off %d, in off %d\n", 
		// 		//     ker, in, input_offset, sum - bias[i_ch_out],
		// 		//     ker_ptr - kernel,
		// 		// 	in_ptr - input
		// 		// );
		// 		printf("\tout is now %d, += %d * %d (input_offset: %d, bias: %d)\n", sum, ker, in + input_offset, input_offset, bias[i_ch_out]);
		// 		fflush(stdout);

		// 		ker_ch_count--;
		// 	}
		// 	printf("\ti sum %d (bias %d)\n", sum, bias[i_ch_out]);
		// 	fflush(stdout);
		// 	sum = arm_nn_requantize(sum, output_mult[i_ch_out],
		// 			output_shift[i_ch_out]);
		// 	sum += out_offset;
		// 	sum = MAX(sum, out_activation_min);
		// 	sum = MIN(sum, out_activation_max);
		// 	*out++ = (int8_t) sum;
		// 	printf("c|x|i %d %d|e %d\n", i_element, i_ch_out, sum);
		// 	fflush(stdout);
		// }
		for (int32_t i_ch_in = 0; i_ch_in < input_ch; i_ch_in++) {
			dst[i_ch_in] = (int16_t)(src[i_ch_in] + input_offset);
		}
#ifndef CLAMPING_PRED_TEST
		out = mat_mult_kernel_s8_s16_one_column_clamping_pred(kernel,
				runtime_buf, termination_steps, termination_bounds,
				output_ch, output_shift, output_mult,
				(int8_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);
#else
		out = mat_mult_kernel_s8_s16_one_column_clamping_pred_with_profile(kernel, 
				runtime_buf, termination_steps, termination_bounds, 
				input_offset, output_ch, output_shift, output_mult,
				(int8_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);
#endif // CLAMPING_PRED_TEST
	}

	/* Return to application */
	return STATE_SUCCESS;
}


tinyengine_status convolve_1x1_s8_uint16_steps_clamping_pred(const int8_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const int8_t *kernel,
		const uint16_t *termination_steps,
		const int32_t *termination_bounds,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, int8_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, int16_t *runtime_buf) {


	(void) input_x;
	(void) input_y;

	int8_t *out = output;
	const int32_t num_elements = output_x * output_y;

	for (int32_t i_element = 0; i_element < num_elements; i_element++) {
		// int32_t i_ch_out;
		// const int8_t *ker_ptr = kernel;
		int8_t *src = &input[i_element * input_ch];
		int16_t *dst = runtime_buf;

		// for (int32_t i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
		// 	int32_t sum = bias[i_ch_out];
		// 	uint16_t ker_ch_count = input_ch * DIM_KER_X * DIM_KER_Y;
		// 	const int8_t *in_ptr = &input[i_element * input_ch];

		// 	while (ker_ch_count) {
		// 		int16_t in = *in_ptr++;
		// 		int8_t ker = *ker_ptr++;
		// 		sum += (in + input_offset) * ker;
		// 		// printf("\t sum += %d * (%d + %d), now %d. ker off %d, in off %d\n", 
		// 		//     ker, in, input_offset, sum - bias[i_ch_out],
		// 		//     ker_ptr - kernel,
		// 		// 	in_ptr - input
		// 		// );
		// 		printf("\tout is now %d, += %d * %d (input_offset: %d, bias: %d)\n", sum, ker, in + input_offset, input_offset, bias[i_ch_out]);
		// 		fflush(stdout);

		// 		ker_ch_count--;
		// 	}
		// 	printf("\ti sum %d (bias %d)\n", sum, bias[i_ch_out]);
		// 	fflush(stdout);
		// 	sum = arm_nn_requantize(sum, output_mult[i_ch_out],
		// 			output_shift[i_ch_out]);
		// 	sum += out_offset;
		// 	sum = MAX(sum, out_activation_min);
		// 	sum = MIN(sum, out_activation_max);
		// 	*out++ = (int8_t) sum;
		// 	printf("c|x|i %d %d|e %d\n", i_element, i_ch_out, sum);
		// 	fflush(stdout);
		// }
		for (int32_t i_ch_in = 0; i_ch_in < input_ch; i_ch_in++) {
			dst[i_ch_in] = (int16_t)(src[i_ch_in] + input_offset);
		}
#ifndef CLAMPING_PRED_TEST
		out = mat_mult_kernel_s8_s16_one_column_uint16_steps_clamping_pred(kernel,
				runtime_buf, termination_steps, termination_bounds,
				output_ch, output_shift, output_mult,
				(int8_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);
#else
		out = mat_mult_kernel_s8_s16_one_column_uint16_steps_clamping_pred_with_profile(kernel, 
				runtime_buf, termination_steps, termination_bounds, 
				input_offset, output_ch, output_shift, output_mult,
				(int8_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);
#endif // CLAMPING_PRED_TEST
	}

	/* Return to application */
	return STATE_SUCCESS;
}