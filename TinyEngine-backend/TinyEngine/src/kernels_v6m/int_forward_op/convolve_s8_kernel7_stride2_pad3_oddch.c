#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status convolve_s8_kernel7_stride2_pad3_oddch(
		const int8_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, 
		const int8_t *kernel, 
		const int32_t *bias,
		const int32_t *output_shift, const int32_t *output_mult,
		const int32_t output_offset, const int32_t input_offset,
		const int32_t output_activation_min,
		const int32_t output_activation_max, int8_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, int16_t *runtime_buf, int8_t pad_value
) {
	(void) input_x;
	(void) input_y;

	/* Partial(two columns) im2col buffer */
	int16_t *input_16b_buffer = runtime_buf;
	int8_t *out = output;

	const int16_t inoff16 = input_offset;
	const int16_t pad_out = inoff16 + pad_value;
	int in_row_offset = input_ch * input_x;

	// Create im2col buffer with padding
	// as pad offset is 1, only padding on the right and lower end
	for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
		const int16_t base_idx_y = (i_out_y * 2) - 3; // stride is 2, pad 3
		// Check y idx to determine y-axis padding
		if (base_idx_y == -3) { // pad three rows
			int16_t *dst = (int16_t*) &input_16b_buffer[input_ch * 0];
			int block_cnt = input_ch;
			while (block_cnt > 0) {//total: 16bit * input_ch * 7 // kernel 7x7
				*dst++ = pad_out; *dst++ = pad_out; *dst++ = pad_out; *dst++ = pad_out; *dst++ = pad_out; *dst++ = pad_out; *dst++ = pad_out;
				block_cnt--;
			}
			for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
				const int16_t base_idx_x = (i_out_x * 2) - 3; // stride 2 pad 3

				if (base_idx_x == -3) { // pad three cols
				} else if (base_idx_x == -2) { // pad two cols
					// Impossible to reach
				} else if (base_idx_x == -1) { // pad one col
				} else if (base_idx_x == input_x - 5) { // pad last one col
				} else if (base_idx_x == input_x - 4) { // pad last two cols
				} else if (base_idx_x == input_x - 3) { // pad last three cols
				}
				out = mat_mult_kernel_s8_s16_one_column(kernel, runtime_buf,
						output_ch, output_shift, output_mult, output_offset,
						output_activation_min, output_activation_max,
						input_ch * 49, // kernel 7x7
						bias, out);
			}
		} else if (base_idx_y == -2) { // pad two rows
			// Impossible to reach
		} else if (base_idx_y == -1) { // pad one row
		} else if (base_idx_y == input_y - 5) { // pad one row in the bottom
		} else if (base_idx_y == input_y - 4) { // pad two rows in the bottom
		} else if (base_idx_y == input_y - 3) { // pad three rows in the bottom
		}

		if (base_idx_y + 2 == input_y) { //pad the third row
			int16_t *dst = (int16_t*) &input_16b_buffer[input_ch * 6];
			int block_cnt = input_ch;
			while (block_cnt > 0) {//total: 16bit * input_ch * 3
				*dst++ = pad_out; *dst++ = pad_out; *dst++ = pad_out;
				block_cnt--;
			}
		} else { // no y-axis pad needed
			for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
				const int16_t base_idx_x = (i_out_x * 2) - 0; // stride 2
				if (base_idx_x + 2 == input_x) {
					/* load 2 col */
					const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
					const int8_t *src2 = src + in_row_offset;
					const int8_t *src3 = src2 + in_row_offset;
					// printf("\tbase_idx_x is %d (i_out_x %d, input_x %d). loading input from %d %d %d and skip last col\n",base_idx_x, i_out_x, input_x, src - input, src2 - input, src3 - input);
					// fflush(stdout);
					int16_t *dst = &input_16b_buffer[0];;
					int16_t *dst2 = &input_16b_buffer[input_ch * 3];
					int16_t *dst3 = &input_16b_buffer[input_ch * 6];;

					int block_cnt = input_ch * 2; // 2 cols
					while (block_cnt > 0) {
						*dst++ = (int16_t)*src++ + inoff16;
						*dst2++ = (int16_t)*src2++ + inoff16;
						*dst3++ = (int16_t)*src3++ + inoff16;
						block_cnt--;
					}

					/* use pad for the last 1 col*/
					block_cnt = input_ch;
					while (block_cnt > 0) {
						*dst++ = pad_out;
						*dst2++ = pad_out;
						*dst3++ = pad_out;
						block_cnt--;
					}
				} else {
					/* load 3 col */
					const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
					const int8_t *src2 = src + in_row_offset;
					const int8_t *src3 = src2 + in_row_offset;
					// printf("\tbase_idx_x is %d (i_out_x %d, input_x %d). loading input from %d %d %d\n",base_idx_x, i_out_x, input_x, src - input, src2 - input, src3 - input);
					// fflush(stdout);
					int16_t *dst = &input_16b_buffer[0];;
					int16_t *dst2 = &input_16b_buffer[input_ch * 3];
					int16_t *dst3 = &input_16b_buffer[input_ch * 6];;
					int block_cnt = input_ch * 3; // 3 cols
					while (block_cnt > 0) {
						// printf("\t\tload %d %d %d\n", *src, *src2, *src3);
						// fflush(stdout);
						*dst++ = (int16_t)*src++ + inoff16;
						*dst2++ = (int16_t)*src2++ + inoff16;
						*dst3++ = (int16_t)*src3++ + inoff16;
						block_cnt--;
					}
				}

				// printf("\toutptr is now %d\n", out - output);
				// fflush(stdout);
			}
		}
	}
	// printf("printing output\n");
	// fflush(stdout);
	// int idx = 0;
	// for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
	// 	for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
	// 		for (int i_out_c = 0; i_out_c < output_ch; i_out_c++) {
	// 			printf("c|x|i %d %d %d|e %d\n", i_out_y, i_out_x, i_out_c, output[idx++]);
	// 			fflush(stdout);
	// 		}
	// 	}
	// }
	// printf("end printing output\n");
	// fflush(stdout);

	/* Return to application */
	return STATE_SUCCESS;
}

/**
 * @} end of NNConv group
 */
