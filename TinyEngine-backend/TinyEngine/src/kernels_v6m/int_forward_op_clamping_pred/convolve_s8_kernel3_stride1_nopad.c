#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

tinyengine_status convolve_s8_kernel3_stride1_nopad_clamping_pred(
        const int8_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, const int8_t *kernel, 
		const uint8_t *termination_steps,
		const int32_t *termination_bounds,
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
            /* load 3 col */
            const int8_t *src = input + (base_idx_y * input_x + base_idx_x) * input_ch;
            const int8_t *src2 = src + in_row_offset;
            const int8_t *src3 = src2 + in_row_offset; // 3 rows
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
#ifndef CLAMPING_PRED_TEST
            out = mat_mult_kernel_s8_s16_one_column_clamping_pred(kernel,
                                    runtime_buf,
                                    termination_steps, 
                                    termination_bounds,
                                    output_ch,
                                    output_shift,
                                    output_mult,
                                    output_offset,
                                    output_activation_min,
                                    output_activation_max,
                                    input_ch * 9,
                                    bias,
                                    out);
#else
            out = mat_mult_kernel_s8_s16_one_column_clamping_pred_with_profile(kernel, 
                                    runtime_buf,
                                    termination_steps, 
                                    termination_bounds,
                                    input_offset,
                                    output_ch,
                                    output_shift,
                                    output_mult,
                                    output_offset,
                                    output_activation_min,
                                    output_activation_max,
                                    input_ch * 9,
                                    bias,
                                    out);
#endif // CLAMPING_PRED_TEST
        }
	}
	/* Return to application */
	return STATE_SUCCESS;
}