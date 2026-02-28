#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status convolve_s8_kernel1_stride2(const int8_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const int8_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, int8_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, int16_t *runtime_buf) {


	(void) input_x;
	(void) input_y;

	int8_t *out = output;
	const int32_t num_elements = output_x * output_y;

	for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
		const int16_t base_idx_y = i_out_y * 2 - 0; // stride 2, pad 0
        for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
            const int16_t base_idx_x = i_out_x * 2 - 0; // stride 2, pad 0
			int8_t *src = &input[(base_idx_y * input_x + base_idx_x) * input_ch];
			int16_t *dst = runtime_buf;

			for (int32_t i_ch_in = 0; i_ch_in < input_ch; i_ch_in++) {
				dst[i_ch_in] = (int16_t)(src[i_ch_in] + input_offset);
			}
			out = mat_mult_kernel_s8_s16_one_column(kernel,
					runtime_buf, output_ch, output_shift, output_mult,
					(int8_t) out_offset, out_activation_min,
					out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
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
