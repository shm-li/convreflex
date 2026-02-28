#include <stdio.h>

#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"
#include "tinyengine_lib.h"

/*
   * Matrix-multiplication function for convolution with uint8
   */
int8_t *mat_mult_kernel_s8_s16_one_column(const int8_t *input_a,
							const int16_t *input_b,
							const uint16_t output_ch,
							const int32_t *out_shift,
							const int32_t *out_mult,
							const int32_t out_offset,
							const int16_t activation_min,
							const int16_t activation_max,
							const uint16_t num_col_a,
							const int32_t *const output_bias,
							int8_t *out_0)
{
    /* This version only takes in one input column from input_b! */
    //int8_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    // uint16_t row_count = output_ch / 2;
    uint32_t row_count = output_ch;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        // const int16_t *ip_b1 = ip_b0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        // int32_t ch_0_out_1 = 0;

        /* load the bias */
        // if (bias)
        // {
            ch_0_out_0 = *bias;
            // ch_0_out_1 = *bias++;
        // }

        uint32_t col_count = num_col_a;
        // while (col_count)
        // {
        //     int8_t a0 = *ip_a0++;
        //     int16_t b0 = *ip_b0++;
        //     // int16_t b1 = *ip_b1++;

        //     ch_0_out_0 += a0 * b0;
        //     // printf("\tout is now %d, += %d * %d (bias: %d)\n", ch_0_out_0, a0, b0, *bias);
        //     // fflush(stdout);
        //     // ch_0_out_1 += a0 * b1;
        //     col_count--;
        // }
        for (int32_t i_col = 0; i_col < col_count; ++i_col) {
            ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
        }
        ip_a0 += num_col_a;
        ip_b0 += num_col_a;
        
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;
        // printf("e %d\n", ch_0_out_0);
        // fflush(stdout);

        // ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        // ch_0_out_1 += out_offset;
        // ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        // ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        // *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;
        bias++;

        row_count--;
    }

    /* return the new output pointer with offset */
    return out_0;
}
