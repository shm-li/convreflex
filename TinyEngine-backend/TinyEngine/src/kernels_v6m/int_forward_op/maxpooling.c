/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   maxpooling.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <stdio.h>
#include "tinyengine_function.h"

tinyengine_status max_pooling(const q7_t* input, const uint16_t input_h, const uint16_t input_w,
		const uint16_t input_c,	const uint16_t sample_h, const uint16_t sample_w,
		const uint16_t output_h, const uint16_t output_w, const int32_t out_activation_min,
        const int32_t out_activation_max, q7_t* output)
{
	int h, w, c;
	int sh, sw;
	for(c = 0; c < input_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				int max = out_activation_min;

				for(sh = 0; sh < sample_h; sh++){
					int height = sh + h * sample_h;
					for(sw = 0; sw < sample_w; sw++){
						int width = sw + w * sample_w;
						max = TN_MAX(max,input[(width + height * input_w) * input_c + c]);
					}
				}

				int out = max;
				out = TN_MAX(out, out_activation_min);
				out = TN_MIN(out, out_activation_max);
				output[(w + h * output_w) * input_c + c] = out;
			}
		}
	}
}


tinyengine_status max_pooling_filternxn_stridenxn_padnxn(const q7_t* input, 
		const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
		const uint16_t filter_len, const uint16_t stride_len, const uint16_t pad_len,
		const uint16_t output_h, const uint16_t output_w, const int32_t out_activation_min,
        const int32_t out_activation_max, q7_t* output, int8_t pad_value)
{
	int h, w, c;
	int sh, sw;
	// printf("printing outputs\n");
	for(h = 0; h < output_h; h++){
		const int base_idx_h = h * stride_len - pad_len;
		int sh_start = 0;
		int sh_end = filter_len;
		int min_val = out_activation_min;
		if (base_idx_h < 0) { // pad on top
			sh_start = -base_idx_h;
			min_val = pad_value;
		} else if (base_idx_h + filter_len > input_h) { // pad on bottom
			sh_end = input_h - base_idx_h;
			min_val = pad_value;
		} 
		int min_val_checkpoint = min_val;
		for(w = 0; w < output_w; w++){
			const int base_idx_w = w * stride_len - pad_len;
			
			int sw_start = 0;
			int sw_end = filter_len;
			min_val = min_val_checkpoint;
			if (base_idx_w < 0) { // pad left
				sw_start = -base_idx_w;
				min_val = pad_value;
			} else if (base_idx_w + filter_len > input_w) { // pad right
				sw_end = input_w - base_idx_w;
				min_val = pad_value;
			} 

			for(c = 0; c < input_c; c++){
				int max = min_val;
				for(sh = sh_start; sh < sh_end; sh++){
					int height = sh + base_idx_h;
					for(sw = sw_start; sw < sw_end; sw++){
						int width = sw + base_idx_w;
						// printf("\t%d %d (%d %d): compare %d %d\n", height, width, base_idx_h, base_idx_w, max, input[(width + height * input_w) * input_c + c]);
						max = TN_MAX(max,input[(width + height * input_w) * input_c + c]);
					}
				}

				int out = max;
				out = TN_MAX(out, out_activation_min);
				out = TN_MIN(out, out_activation_max);
				output[(w + h * output_w) * input_c + c] = out;
				// printf("MAX %d %d %d (%d): %d\n", h, w, c, (w + h * output_w) * input_c + c, out);
				// fflush(stdout);
			}
		}
	}
	// printf("end printing outputs\n");
	// fflush(stdout);
}