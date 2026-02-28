#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status convolve_s8_kernelnxn_stridenxn_padnxn(
		const int8_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, 
		const uint16_t kernel_n,
		const uint16_t stride_n,
		const uint16_t pad_n,
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

	const int elem_per_row = kernel_n * input_ch; // total: kernel_w * input_ch

	// Create im2col buffer with padding
	for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
		const int16_t base_idx_y = (i_out_y * stride_n) - pad_n;
		const int pad_top = MAX(0, 0 - base_idx_y);
		const int pad_bottom = MAX(0, base_idx_y + (kernel_n) - input_y);
		const int no_pad_rows = kernel_n - pad_top - pad_bottom;

		const int src_start_y = base_idx_y + pad_top;

		// Create im2col buffer and pad top
		int16_t *dst = (int16_t*) &input_16b_buffer[input_ch * 0];
		int elem_cnt = elem_per_row * pad_top;
		while (elem_cnt > 0) { //total: 16bit * input_ch * kernel_w * pad_top
			*dst++ = pad_out;
			elem_cnt--;
		}
		// Pad bottom
		int16_t* dst_end = dst + elem_per_row * no_pad_rows;
		elem_cnt = elem_per_row * pad_bottom;
		while (elem_cnt > 0) {
			*dst_end++ = pad_out;
			elem_cnt--;
		}

		const int16_t* cur_dst = dst;
		for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
			dst = cur_dst;
			const int16_t base_idx_x = (i_out_x * stride_n) - pad_n;
			const int pad_left = MAX(0, 0 - base_idx_x);
			const int pad_right = MAX(0, base_idx_x + (kernel_n) - input_x);
			const int no_pad_cols = kernel_n - pad_left - pad_right;

			const int src_start_x = base_idx_x + pad_left;
			const int8_t *src = input + (src_start_y * input_x + src_start_x) * input_ch;

			int row_cnt = no_pad_rows;
			while (row_cnt > 0) {
				// Pad left
				elem_cnt = input_ch * pad_left;
				while (elem_cnt > 0) {
					*dst++ = pad_out;
					elem_cnt--;
				}
				elem_cnt = input_ch * no_pad_cols;
				int src_idx = 0;
				while (src_idx < elem_cnt) {
					*dst++ = (int16_t)src[src_idx] + inoff16;
					src_idx++;
				}
				// Pad right
				elem_cnt = input_ch * pad_right;
				while (elem_cnt > 0) {
					*dst++ = pad_out;
					elem_cnt--;
				}
				src += in_row_offset;
				row_cnt--;
			}

			out = mat_mult_kernel_s8_s16_one_column(kernel, runtime_buf,
					output_ch, output_shift, output_mult, output_offset,
					output_activation_min, output_activation_max,
					input_ch * kernel_n * kernel_n,
					bias, out);
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
