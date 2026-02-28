#include <stdio.h>

// #include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status convolve_s8_kernelnx1_stride1_nopad(
        const int8_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, 
        const uint16_t kernel_y, const int8_t *kernel, 
        const int32_t *bias,
		const int32_t *output_shift, const int32_t *output_mult,
		const int32_t output_offset, const int32_t input_offset,
		const int32_t output_activation_min,
		const int32_t output_activation_max, int8_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, int16_t *runtime_buf) {
	(void) input_x;
	(void) input_y;

	int16_t *input_16b_buffer = runtime_buf;
	int8_t *out = output;

	const int16_t inoff16 = input_offset;
	int in_row_offset = input_ch * input_x;

	// Create im2col buffer
	for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
		const int16_t base_idx_y = i_out_y - 0; // pad 0
        for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
            const int16_t base_idx_x = i_out_x - 0; // pad 0
            /* load n rows */

            const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
            // const int8_t *src2 = src + in_row_offset;
            int16_t *dst = &input_16b_buffer[0];;
            // int16_t *dst2 = &input_16b_buffer[input_ch * 1];
            int ker_y_cnt = kernel_y;
            while (ker_y_cnt > 0) {
                int block_cnt = input_ch; // 1 col, repeat kernel_y (row) times
                const int8_t *src_ptr = src;
                // int16_t *dst_ptr = dst;
                while (block_cnt > 0) {
                    *dst++ = (int16_t)*src_ptr++ + inoff16;
                    block_cnt--;
                }
                src += in_row_offset;
                // dst += input_ch;
                ker_y_cnt--;
            }

            out = mat_mult_kernel_s8_s16_one_column(kernel,
                                    runtime_buf,
                                    output_ch,
                                    output_shift,
                                    output_mult,
                                    output_offset,
                                    output_activation_min,
                                    output_activation_max,
                                    input_ch * kernel_y,
                                    bias,
                                    out);
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