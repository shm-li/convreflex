#include <stdio.h>

// #include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status convolve_s8_kernel3_stride2_padoffset1(
		const int8_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, const int8_t *kernel, const int32_t *bias,
		const int32_t *output_shift, const int32_t *output_mult,
		const int32_t output_offset, const int32_t input_offset,
		const int32_t output_activation_min,
		const int32_t output_activation_max, int8_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, int16_t *runtime_buf, int8_t pad_value
) {

	if (input_ch % 4 != 0 || input_ch % 2 != 0) {
		printf("input_ch %d not divisible by 4\n", input_ch);
		fflush(stdout);
		return PARAM_NO_SUPPORT;
	}

	(void) input_x;
	(void) input_y;

	/* Partial(two columns) im2col buffer */
	int16_t *input_16b_buffer = runtime_buf;
	int8_t *out = output;
	const int channel_div4 = (input_ch >> 2);
	const int channel_div2 = (input_ch >> 1);

	const int16_t inoff16 = input_offset;
	const int32_t pad_out = input_offset + pad_value;
	int32_t pad_out_q15x2 = pad_out & (pad_out << 16);
	int8_t* bytes = &pad_out_q15x2;
	// printf("CHECKING pad_out_q15x2: %d, %x, %x, %x, %x\n", pad_out_q15x2, bytes[0], bytes[1], bytes[2], bytes[3]);
	int in_row_offset = input_ch * input_x;

	// Create im2col buffer with padding
	// as pad offset is 1, only padding on the right and lower end
	for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
		const int16_t base_idx_y = (i_out_y * 2) - 0; // stride is 2, pad 0
		// Check y idx to determine y-axis padding
		if (base_idx_y + 2 == input_y) { //pad the third row
			int32_t *dst_32 = (int32_t*) &input_16b_buffer[input_ch * 6];
			int block_cnt = channel_div4;//unroll by 2, 3 element
			while (block_cnt > 0) {//total: 16bit * input_ch * 3
				*dst_32++ = pad_out_q15x2; *dst_32++ = pad_out_q15x2;
				*dst_32++ = pad_out_q15x2; *dst_32++ = pad_out_q15x2;
				*dst_32++ = pad_out_q15x2; *dst_32++ = pad_out_q15x2;
				block_cnt--;
			}
			for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
				const int16_t base_idx_x = (i_out_x * 2) - 0; // stride 2, pad 0
				if (base_idx_x + 2 == input_x) {
					/* load 2 col*/
					int16_t *dst = &input_16b_buffer[input_ch * 0];
					int16_t *dst2  = &input_16b_buffer[input_ch * 3];
					const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
					const int8_t *src2 = src + in_row_offset;

					int block_cnt = input_ch * 2; // 2 cols
					while (block_cnt > 0) {
						*dst++ = (int16_t)*src++ + inoff16;
						*dst2++ = (int16_t)*src2++ + inoff16;
						block_cnt--;
					}

					/* use pad for the last 1 col*/
					int32_t *dst_32 = (int32_t*) dst;
					int32_t *dst2_32 = (int32_t*) dst2;

					block_cnt = channel_div4;
					while (block_cnt > 0) {
						*dst_32++ = pad_out_q15x2; *dst_32++ = pad_out_q15x2;
						*dst2_32++ = pad_out_q15x2; *dst2_32++ = pad_out_q15x2;
						block_cnt--;
					}
				} else {
					/* load 3 col*/
					int16_t *dst = &input_16b_buffer[input_ch * 0];
					int16_t *dst2  = &input_16b_buffer[input_ch * 3];
					/* load input to 1 col*/
					const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
					const int8_t *src2 = src + in_row_offset;

					int block_cnt = input_ch * 3; // 3 cols
					while (block_cnt > 0) {
						*dst++ = (int16_t)*src++ + inoff16;
						*dst2++ = (int16_t)*src2++ + inoff16;
						block_cnt--;
					}
				}
				out = mat_mult_kernel_s8_s16_one_column(kernel,
										runtime_buf,
										output_ch,
										output_shift,
										output_mult,
										output_offset,
										output_activation_min,
										output_activation_max,
										input_ch * 9,
										bias,
										out);
			}
		} else { // no y-axis pad needed
			for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
				const int16_t base_idx_x = (i_out_x * 2) - 0; // stride 2, pad 0
				if (base_idx_x + 2 == input_x) {
					/* load 2 col */
					const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
					const int8_t *src2 = src + in_row_offset;
					const int8_t *src3 = src2 + in_row_offset;
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
					int32_t *dst_32 = (int32_t*) dst;
					int32_t *dst2_32 = (int32_t*) dst2;
					int32_t *dst3_32 = (int32_t*) dst3;
					block_cnt = channel_div4;
					while (block_cnt > 0) {
						*dst_32++ = pad_out_q15x2; *dst_32++ = pad_out_q15x2;
						*dst2_32++ = pad_out_q15x2; *dst2_32++ = pad_out_q15x2;
						*dst3_32++ = pad_out_q15x2; *dst3_32++ = pad_out_q15x2;
						block_cnt--;
					}
				} else {
					/* load 3 col */
					const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
					const int8_t *src2 = src + in_row_offset;
					const int8_t *src3 = src2 + in_row_offset;
					int16_t *dst = &input_16b_buffer[0];;
					int16_t *dst2 = &input_16b_buffer[input_ch * 3];
					int16_t *dst3 = &input_16b_buffer[input_ch * 6];;
					int block_cnt = input_ch * 3; // 3 cols
					while (block_cnt > 0) {
						*dst++ = (int16_t)*src++ + inoff16;
						*dst2++ = (int16_t)*src2++ + inoff16;
						*dst3++ = (int16_t)*src3++ + inoff16;
						block_cnt--;
					}
				}

				out = mat_mult_kernel_s8_s16_one_column(kernel,
										runtime_buf,
										output_ch,
										output_shift,
										output_mult,
										output_offset,
										output_activation_min,
										output_activation_max,
										input_ch * 9,
										bias,
										out);
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
