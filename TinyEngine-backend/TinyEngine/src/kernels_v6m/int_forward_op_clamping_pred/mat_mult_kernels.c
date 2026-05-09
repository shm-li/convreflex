#include <stdio.h>

#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)

#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"
#include "tinyengine_lib.h"

int8_t *mat_mult_kernel_s8_s16_one_column_clamping_pred(const int8_t *input_a,
							const int16_t *input_b,
                            const uint8_t *termination_steps,
                            const int32_t *termination_bounds,
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
    const int8_t *start_out_0 = out_0;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    for (int32_t ch = 0; ch < output_ch; ++ch)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        // const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* load the bias */
        int32_t ch_0_out_0 = *bias;

        const int32_t termination_check_remaining = *termination_steps++;
        int32_t col_count = num_col_a;
        if (termination_check_remaining == 1) {
            int32_t next = *termination_steps++;
            int32_t i_col = 0;
            int32_t termination_type = termination_bounds[0]; // we store the type here
                                                            // we only consider type == 1 (Underflow)

            for ( ; i_col < next; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
            }
            // int8_t* tmp_ip_a0 = ip_a0;
            // int16_t* tmp_ip_b0 = ip_b0;
            // uint32_t remaining_col = next;
            // while (remaining_col)
            // {
            //     int8_t a0 = *tmp_ip_a0++;
            //     int16_t b0 = *tmp_ip_b0++;

            //     ch_0_out_0 += a0 * b0;
            //     remaining_col--;
            // }
            /**
             * { Check for early-termination
             */
            // if ((termination_type == 1) && (ch_0_out_0 < termination_bounds[1])) {
            if (unlikely(ch_0_out_0 < termination_bounds[1])) {
                // Underflow
                termination_bounds += 2;

                ip_a0 += num_col_a;
                // ip_b0 = input_b;
                // ip_a0 = last_ip_a0 + num_col_a;

                ch_0_out_0 = activation_min;
                goto terminate_neuron;
            // } else if ((termination_type == 2) && (ch_0_out_0 > termination_bounds[1])) {
            //     // Overflow
            //     termination_bounds += 2;
            //     ip_a0 += num_col_a;
            //     ch_0_out_0 = activation_max;
            //     goto terminate_neuron;
            // }
            } else {
                for ( ; i_col < num_col_a; ++i_col) {
                    ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                }
                termination_bounds += 2;
                ip_a0 += num_col_a; // need to increment this when using redirection

                // int8_t* tmp_ip_a0 = ip_a0 + next;
                // int16_t* tmp_ip_b0 = ip_b0 + next;
                // remaining_col = num_col_a - next;
                // while (remaining_col)
                // {
                //     int8_t a0 = *tmp_ip_a0++;
                //     int16_t b0 = *tmp_ip_b0++;

                //     ch_0_out_0 += a0 * b0;
                //     remaining_col--;
                // }
                // termination_bounds += 2;
                // ip_a0 += num_col_a;
            }
        } else if (termination_check_remaining == 0) {
        // } else {
            for (int32_t i_col = 0; i_col < num_col_a; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
            }
            ip_a0 += num_col_a; // need to increment this when using redirection

            // int8_t* tmp_ip_a0 = ip_a0;
            // int16_t* tmp_ip_b0 = ip_b0;
            // uint32_t remaining_col = num_col_a;
            // while (remaining_col)
            // {
            //     int8_t a0 = *tmp_ip_a0++;
            //     int16_t b0 = *tmp_ip_b0++;

            //     ch_0_out_0 += a0 * b0;
            //     remaining_col--;
            // }
            // ip_a0 += num_col_a;
        }
        
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);

terminate_neuron:
        *out_0++ = (int8_t)ch_0_out_0;

        out_mult++;
        out_shift++;
        bias++;
    }

    /* return the new output pointer with offset */
    return out_0;
}


int8_t *mat_mult_kernel_s8_s16_one_column_clamping_pred_with_profile(const int8_t *input_a,
							const int16_t *input_b,
                            const uint8_t *termination_steps,
                            const int32_t *termination_bounds,
                            const int32_t in_offset,
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
    const int8_t *start_out_0 = out_0;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    for (int32_t ch = 0; ch < output_ch; ++ch)
    {
        // For profiling
        int termination_type = 0; // 0: none; 1: underflow; 2: overflow
        int termination_step = 0;
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        // const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* load the bias */
        int32_t ch_0_out_0 = *bias;

        const int32_t termination_check_remaining = *termination_steps++;
        int32_t col_count = num_col_a;
        if (termination_check_remaining == 1) {
            int32_t next = *termination_steps++;
            int32_t i_col = 0;
            int32_t termination_check_type = termination_bounds[0]; // we store the type here
            for ( ; i_col < next; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                // printf("\t %d * %d, now %d %d\r\n", ip_a0[i_col], ip_b0[i_col], ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset); fflush(stdout);
            }
            /**
             * { Check for early-termination
             */
            if ((termination_check_type == 1) && (ch_0_out_0 < termination_bounds[1])) {
                // printf("\t\t%d: ch_0_out_0 %d (%d) < bound %d (%d)\r\n", 
                //     i_col,
                //     ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset, 
                //     termination_bounds[1], arm_nn_requantize(termination_bounds[1], *out_mult, *out_shift) + out_offset);
                // Underflow
                termination_type = 1;
                termination_step = i_col;

                termination_bounds += 2;
                ip_a0 += num_col_a;
                ch_0_out_0 = activation_min;
                goto terminate_neuron_profile;
            } else if ((termination_check_type == 2) && (ch_0_out_0 > termination_bounds[1])) {
                // Overflow
                termination_type = 2;
                termination_step = i_col;

                termination_bounds += 2;
                ip_a0 += num_col_a;
                ch_0_out_0 = activation_max;
                goto terminate_neuron_profile;
            }
            for ( ; i_col < num_col_a; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                // printf("\t %d * %d, now %d %d\r\n", ip_a0[i_col], ip_b0[i_col], ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset); fflush(stdout);
            }
            termination_bounds += 2;
            ip_a0 += num_col_a; // need to increment this when using redirection
        } else if (termination_check_remaining == 0) {
        // } else {
            for (int32_t i_col = 0; i_col < num_col_a; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                // printf("\t %d * %d, now %d %d\r\n", ip_a0[i_col], ip_b0[i_col], ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset); fflush(stdout);
            }
            ip_a0 += num_col_a; // need to increment this when using redirection
        }
        
        // printf("\t RESULT: %d, check: %d\r\n", ch_0_out_0, termination_check_remaining);
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);

terminate_neuron_profile:

        /**
         * { At the end, re-do computations for profiling
         */
        // for profiling
        int32_t clamping_start_step = 0;
        int32_t false_clamping_count = 0;
        // bool clamping = false;
        int32_t clamping_type = 0; // 0: none; 1: underflow; 2: overflow
        int32_t safely_omittable_computation_start_step = 0;
        bool already_safely_omittable = false;

        // printf("\tout = %d. \r\n", ch_0_out_0);
        // fflush(stdout);
        int32_t temp_col_count = num_col_a;
        int32_t temp_out = *bias;
        int8_t* temp_ip_a0 = ip_a0 - num_col_a;
        int16_t* temp_ip_b0 = ip_b0;
        int32_t acc_change[num_col_a + 1];
        acc_change[0] = arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset;
        while (temp_col_count) {
            temp_out += *temp_ip_a0++ * *temp_ip_b0++;
            // printf("\t %d * %d, verif now %d %d\r\n", *(temp_ip_a0-1), *(temp_ip_b0-1), temp_out, arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset); fflush(stdout);
            temp_col_count--;

            int32_t quant_temp_out = arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset;
            // Profile where effectless computations start
            if (quant_temp_out <= activation_min || (quant_temp_out >= activation_max)) {
                // record clamping now
                if (clamping_type == 0) {
                    clamping_start_step = num_col_a - temp_col_count;
                    if (quant_temp_out <= activation_min) {
                        clamping_type = 1;
                    } else {
                        clamping_type = 2;
                    }
                }
            } else {
                // Unset "clamping" flag
                if (clamping_type != 0) {
                    false_clamping_count++;
                    clamping_type = 0;
                }
            }
            // Record trace of accumulated value
            acc_change[num_col_a - temp_col_count] = quant_temp_out;
            // Profile safely omittable computations
            // if ((!already_safely_omittable) && (clamping_type != 0)) {
            if (clamping_type != 0) {
                int32_t final_max_out = temp_out;
                int32_t final_min_out = temp_out;
                int32_t final_max_inc = 0;
                int32_t final_min_dec = 0;
                int32_t remaining_col_count = temp_col_count;
                int8_t* temp_temp_ip_a0 = temp_ip_a0;
                while (remaining_col_count) {
                    int8_t a0 = *temp_temp_ip_a0++;
                    final_min_dec += a0 * (a0 > 0 ? (-128 + in_offset) : (127 + in_offset));
                    final_max_inc += a0 * (a0 > 0 ? (127 + in_offset) : (-128 + in_offset));
                    // printf("\t\t\tremaining: final min %d, final max %d. just used weight %d\r\n", final_min_out, final_max_out, a0);
                    // fflush(stdout);
                    remaining_col_count--;
                }
                final_max_out += final_max_inc;
                final_min_out += final_min_dec;
                if (clamping_type == 1) { // Underflow
                    final_max_out = arm_nn_requantize(final_max_out, *out_mult, *out_shift) + out_offset;
                    // printf("\t\t\tquantized: final max %d\r\n", final_max_out);
                    // fflush(stdout);
                    if (final_max_out <= activation_min) {
                        // Fix trace of accumulated value
                        int32_t quant_max_inc = arm_nn_requantize(final_max_inc, *out_mult, *out_shift) + out_offset;
                        // printf("\t fix: %d, %d %d\r\n", activation_min, quant_max_inc, final_max_inc);
                        acc_change[num_col_a - temp_col_count] = activation_min - quant_max_inc;
                        if (quant_temp_out > activation_min - quant_max_inc) {
                            printf("ERROR in calculation\r\n"); fflush(stdout);
                        }
                        if (!already_safely_omittable) {
                            already_safely_omittable = true;
                            safely_omittable_computation_start_step = num_col_a - temp_col_count;
                        }
                    }
                } else if (clamping_type == 2) { // Overflow
                    final_min_out = arm_nn_requantize(final_min_out, *out_mult, *out_shift) + out_offset;
                    // printf("\t\t\tquantized: final min %d\r\n", final_min_out);
                    // fflush(stdout);
                    if (final_min_out >= activation_max) {
                        int32_t quant_min_dec = arm_nn_requantize(final_min_dec, *out_mult, *out_shift) + out_offset;
                        // printf("\t fix: %d, %d %d\r\n", activation_max, quant_min_dec, final_min_dec);
                        acc_change[num_col_a - temp_col_count] = activation_max - quant_min_dec;
                        if (quant_temp_out < activation_max - quant_min_dec) {
                            printf("ERROR in calculation\r\n"); fflush(stdout);
                        }
                        if (!already_safely_omittable) {
                            already_safely_omittable = true;
                            safely_omittable_computation_start_step = num_col_a - temp_col_count;
                        }
                    }
                }
            }
        }
        // printf("\tFinal should be %d->%d, got %d\r\n", temp_out, arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset, ch_0_out_0);
        // fflush(stdout);

        if (termination_type == 0) {
            termination_step = num_col_a;
        }
        if (clamping_type == 0) {
            clamping_start_step = num_col_a;
        }
        if (!already_safely_omittable) {
            safely_omittable_computation_start_step = num_col_a;
        }
        // printf(
        //     "channel %d|termination_type %d|termination_step %d|total_step %d"
        //     "|clamping_type %d|clamping_start %d|false_clamping %d"
        //     "|is_safely_omittable %d|safely_omittable_start %d"
        //     "|non_clamped_out %d|clamped_out %d\r\n",
        //     ch, termination_type, termination_step, num_col_a,
        //     clamping_type, clamping_start_step, false_clamping_count,
        //     already_safely_omittable, safely_omittable_computation_start_step, 
        //     temp_out, ch_0_out_0
        // );
        uint32_t ch = output_ch - row_count;
        printf(
            "ch %d|%d|%d|%d"
            "|%d|%d|%d"
            "|%d|%d"
            "|%d|%d\r\n",
            ch, termination_type, termination_step, num_col_a,
            clamping_type, clamping_start_step, false_clamping_count,
            already_safely_omittable, safely_omittable_computation_start_step, 
            temp_out, ch_0_out_0
        );
#ifndef NO_TRACE_PRINT
        printf("tr %d|%d|%d|", ch, clamping_type, ch_0_out_0);
        for (int i = 0; i < num_col_a; ++i) {
            printf("%d|", acc_change[i]);
        }
        printf("%d\r\n", acc_change[num_col_a]);
        fflush(stdout);
#endif // NO_TRACE_PRINT
        temp_out = arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset;
        temp_out = MAX(temp_out, activation_min);
        temp_out = MIN(temp_out, activation_max);
        // Correctness check
        if (temp_out != ch_0_out_0) {
            printf("\tERROR %d %d\r\n", temp_out, ch_0_out_0);
            fflush(stdout);
        }
        /**
         * }
         */
        *out_0++ = (int8_t)ch_0_out_0;
        // printf("e %d\n", ch_0_out_0);
        // fflush(stdout);

        out_mult++;
        out_shift++;
        bias++;

        row_count--;
    }

    /* return the new output pointer with offset */
    return out_0;
}



int8_t *mat_mult_kernel_s8_s16_one_column_uint16_steps_clamping_pred(const int8_t *input_a,
							const int16_t *input_b,
                            const uint16_t *termination_steps,
                            const int32_t *termination_bounds,
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
    const int8_t *start_out_0 = out_0;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    for (int32_t ch = 0; ch < output_ch; ++ch)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        // const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* load the bias */
        int32_t ch_0_out_0 = *bias;

        const int32_t termination_check_remaining = *termination_steps++;
        int32_t col_count = num_col_a;
        if (termination_check_remaining == 1) {
            int32_t next = *termination_steps++;
            int32_t i_col = 0;
            int32_t termination_type = termination_bounds[0]; // we store the type here
                                                            // we only consider type == 1 (Underflow)

            for ( ; i_col < next; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
            }

            // int8_t* tmp_ip_a0 = ip_a0;
            // int16_t* tmp_ip_b0 = ip_b0;
            // uint32_t remaining_col = next;
            // while (remaining_col)
            // {
            //     int8_t a0 = *tmp_ip_a0++;
            //     int16_t b0 = *tmp_ip_b0++;

            //     ch_0_out_0 += a0 * b0;
            //     remaining_col--;
            // }
            /**
             * { Check for early-termination
             */
            // if ((termination_type == 1) && (ch_0_out_0 < termination_bounds[1])) {
            if (unlikely(ch_0_out_0 < termination_bounds[1])) {
                // Underflow
                termination_bounds += 2;

                ip_a0 += num_col_a;

                // ip_b0 = input_b;
                // ip_a0 = last_ip_a0 + num_col_a;

                ch_0_out_0 = activation_min;
                goto terminate_neuron;
            // } else if ((termination_type == 2) && (ch_0_out_0 > termination_bounds[1])) {
            //     // Overflow
            //     termination_bounds += 2;
            //     ip_a0 += num_col_a;
            //     ch_0_out_0 = activation_max;
            //     goto terminate_neuron;
            // }
            } else {
                for ( ; i_col < num_col_a; ++i_col) {
                    ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                }
                termination_bounds += 2;
                ip_a0 += num_col_a; // need to increment this when using redirection

                // int8_t* tmp_ip_a0 = ip_a0 + next;
                // int16_t* tmp_ip_b0 = ip_b0 + next;
                // remaining_col = num_col_a - next;
                // while (remaining_col)
                // {
                //     int8_t a0 = *tmp_ip_a0++;
                //     int16_t b0 = *tmp_ip_b0++;

                //     ch_0_out_0 += a0 * b0;
                //     remaining_col--;
                // }

                // termination_bounds += 2;
                // ip_a0 += num_col_a;
            }

        } else if (termination_check_remaining == 0) {
        // } else {
            for (int32_t i_col = 0; i_col < num_col_a; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
            }
            ip_a0 += num_col_a; // need to increment this when using redirection

            // int8_t* tmp_ip_a0 = ip_a0;
            // int16_t* tmp_ip_b0 = ip_b0;
            // uint32_t remaining_col = num_col_a;
            // while (remaining_col)
            // {
            //     int8_t a0 = *tmp_ip_a0++;
            //     int16_t b0 = *tmp_ip_b0++;

            //     ch_0_out_0 += a0 * b0;
            //     remaining_col--;
            // }
            // ip_a0 += num_col_a;
        }
        
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);

terminate_neuron:
        *out_0++ = (int8_t)ch_0_out_0;

        out_mult++;
        out_shift++;
        bias++;
    }

    /* return the new output pointer with offset */
    return out_0;
}


int8_t *mat_mult_kernel_s8_s16_one_column_uint16_steps_clamping_pred_with_profile(const int8_t *input_a,
							const int16_t *input_b,
                            const uint16_t *termination_steps,
                            const int32_t *termination_bounds,
                            const int32_t in_offset,
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
    const int8_t *start_out_0 = out_0;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    for (int32_t ch = 0; ch < output_ch; ++ch)
    {
        // For profiling
        int termination_type = 0; // 0: none; 1: underflow; 2: overflow
        int termination_step = 0;
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        // const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* load the bias */
        int32_t ch_0_out_0 = *bias;

        const int32_t termination_check_remaining = *termination_steps++;
        int32_t col_count = num_col_a;
        if (termination_check_remaining == 1) {
            int32_t next = *termination_steps++;
            int32_t i_col = 0;
            int32_t termination_check_type = termination_bounds[0]; // we store the type here
            for ( ; i_col < next; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                // printf("\t %d * %d, now %d %d\r\n", ip_a0[i_col], ip_b0[i_col], ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset); fflush(stdout);
            }
            /**
             * { Check for early-termination
             */
            if ((termination_check_type == 1) && (ch_0_out_0 < termination_bounds[1])) {
                // Underflow
                termination_type = 1;
                termination_step = i_col;

                termination_bounds += 2;
                ip_a0 += num_col_a;
                ch_0_out_0 = activation_min;
                goto terminate_neuron_profile;
            } else if ((termination_check_type == 2) && (ch_0_out_0 > termination_bounds[1])) {
                // Overflow
                termination_type = 2;
                termination_step = i_col;

                termination_bounds += 2;
                ip_a0 += num_col_a;
                ch_0_out_0 = activation_max;
                goto terminate_neuron_profile;
            }
            for ( ; i_col < num_col_a; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                // printf("\t %d * %d, now %d %d\r\n", ip_a0[i_col], ip_b0[i_col], ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset); fflush(stdout);
            }
            termination_bounds += 2;
            ip_a0 += num_col_a; // need to increment this when using redirection
        } else if (termination_check_remaining == 0) {
        // } else {
            for (int32_t i_col = 0; i_col < num_col_a; ++i_col) {
                ch_0_out_0 += ip_a0[i_col] * ip_b0[i_col];
                // printf("\t %d * %d, now %d %d\r\n", ip_a0[i_col], ip_b0[i_col], ch_0_out_0, arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift) + out_offset); fflush(stdout);
            }
            ip_a0 += num_col_a; // need to increment this when using redirection
        }
        
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);

terminate_neuron_profile:

        /**
         * { At the end, re-do computations for profiling
         */
        // for profiling
        int32_t clamping_start_step = 0;
        int32_t false_clamping_count = 0;
        // bool clamping = false;
        int32_t clamping_type = 0; // 0: none; 1: underflow; 2: overflow
        int32_t safely_omittable_computation_start_step = 0;
        bool already_safely_omittable = false;

        // printf("\tout = %d. \r\n", ch_0_out_0);
        // fflush(stdout);
        int32_t temp_col_count = num_col_a;
        int32_t temp_out = *bias;
        int8_t* temp_ip_a0 = ip_a0 - num_col_a;
        int16_t* temp_ip_b0 = ip_b0;
        int32_t acc_change[num_col_a + 1];
        acc_change[0] = arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset;
        while (temp_col_count) {
            temp_out += *temp_ip_a0++ * *temp_ip_b0++;
            // printf("\t %d * %d, verif now %d %d\r\n", *(temp_ip_a0-1), *(temp_ip_b0-1), temp_out, arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset); fflush(stdout);
            temp_col_count--;

            int32_t quant_temp_out = arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset;
            // Profile where effectless computations start
            if (quant_temp_out <= activation_min || (quant_temp_out >= activation_max)) {
                // record clamping now
                if (clamping_type == 0) {
                    clamping_start_step = num_col_a - temp_col_count;
                    if (quant_temp_out <= activation_min) {
                        clamping_type = 1;
                    } else {
                        clamping_type = 2;
                    }
                }
            } else {
                // Unset "clamping" flag
                if (clamping_type != 0) {
                    false_clamping_count++;
                    clamping_type = 0;
                }
            }
            // Record trace of accumulated value
            acc_change[num_col_a - temp_col_count] = quant_temp_out;
            // Profile safely omittable computations
            // if ((!already_safely_omittable) && (clamping_type != 0)) {
            if (clamping_type != 0) {
                int32_t final_max_out = temp_out;
                int32_t final_min_out = temp_out;
                int32_t final_max_inc = 0;
                int32_t final_min_dec = 0;
                int32_t remaining_col_count = temp_col_count;
                int8_t* temp_temp_ip_a0 = temp_ip_a0;
                while (remaining_col_count) {
                    int8_t a0 = *temp_temp_ip_a0++;
                    final_min_dec += a0 * (a0 > 0 ? (-128 + in_offset) : (127 + in_offset));
                    final_max_inc += a0 * (a0 > 0 ? (127 + in_offset) : (-128 + in_offset));
                    // printf("\t\t\tremaining: final min %d, final max %d. just used weight %d\r\n", final_min_out, final_max_out, a0);
                    // fflush(stdout);
                    remaining_col_count--;
                }
                final_max_out += final_max_inc;
                final_min_out += final_min_dec;
                if (clamping_type == 1) { // Underflow
                    final_max_out = arm_nn_requantize(final_max_out, *out_mult, *out_shift) + out_offset;
                    // printf("\t\t\tquantized: final max %d\r\n", final_max_out);
                    // fflush(stdout);
                    if (final_max_out <= activation_min) {
                        // Fix trace of accumulated value
                        int32_t quant_max_inc = arm_nn_requantize(final_max_inc, *out_mult, *out_shift) + out_offset;
                        // printf("\t fix: %d, %d %d\r\n", activation_min, quant_max_inc, final_max_inc);
                        acc_change[num_col_a - temp_col_count] = activation_min - quant_max_inc;
                        if (quant_temp_out > activation_min - quant_max_inc) {
                            printf("ERROR in calculation\r\n"); fflush(stdout);
                        }
                        if (!already_safely_omittable) {
                            already_safely_omittable = true;
                            safely_omittable_computation_start_step = num_col_a - temp_col_count;
                        }
                    }
                } else if (clamping_type == 2) { // Overflow
                    final_min_out = arm_nn_requantize(final_min_out, *out_mult, *out_shift) + out_offset;
                    // printf("\t\t\tquantized: final min %d\r\n", final_min_out);
                    // fflush(stdout);
                    if (final_min_out >= activation_max) {
                        int32_t quant_min_dec = arm_nn_requantize(final_min_dec, *out_mult, *out_shift) + out_offset;
                        // printf("\t fix: %d, %d %d\r\n", activation_max, quant_min_dec, final_min_dec);
                        acc_change[num_col_a - temp_col_count] = activation_max - quant_min_dec;
                        if (quant_temp_out < activation_max - quant_min_dec) {
                            printf("ERROR in calculation\r\n"); fflush(stdout);
                        }
                        if (!already_safely_omittable) {
                            already_safely_omittable = true;
                            safely_omittable_computation_start_step = num_col_a - temp_col_count;
                        }
                    }
                }
            }
        }
        // printf("\tFinal should be %d->%d, got %d\r\n", temp_out, arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset, ch_0_out_0);
        // fflush(stdout);

        if (termination_type == 0) {
            termination_step = num_col_a;
        }
        if (clamping_type == 0) {
            clamping_start_step = num_col_a;
        }
        if (!already_safely_omittable) {
            safely_omittable_computation_start_step = num_col_a;
        }
        // printf(
        //     "channel %d|termination_type %d|termination_step %d|total_step %d"
        //     "|clamping_type %d|clamping_start %d|false_clamping %d"
        //     "|is_safely_omittable %d|safely_omittable_start %d"
        //     "|non_clamped_out %d|clamped_out %d\r\n",
        //     ch, termination_type, termination_step, num_col_a,
        //     clamping_type, clamping_start_step, false_clamping_count,
        //     already_safely_omittable, safely_omittable_computation_start_step, 
        //     temp_out, ch_0_out_0
        // );
        uint32_t ch = output_ch - row_count;
        printf(
            "ch %d|%d|%d|%d"
            "|%d|%d|%d"
            "|%d|%d"
            "|%d|%d\r\n",
            ch, termination_type, termination_step, num_col_a,
            clamping_type, clamping_start_step, false_clamping_count,
            already_safely_omittable, safely_omittable_computation_start_step, 
            temp_out, ch_0_out_0
        );
#ifndef NO_TRACE_PRINT
        printf("tr %d|%d|%d|", ch, clamping_type, ch_0_out_0);
        for (int i = 0; i < num_col_a; ++i) {
            printf("%d|", acc_change[i]);
        }
        printf("%d\r\n", acc_change[num_col_a]);
        fflush(stdout);
#endif // NO_TRACE_PRINT
        temp_out = arm_nn_requantize(temp_out, *out_mult, *out_shift) + out_offset;
        temp_out = MAX(temp_out, activation_min);
        temp_out = MIN(temp_out, activation_max);
        // Correctness check
        if (temp_out != ch_0_out_0) {
            printf("\tERROR %d %d\r\n", temp_out, ch_0_out_0);
            fflush(stdout);
        }
        /**
         * }
         */
        *out_0++ = (int8_t)ch_0_out_0;
        // printf("e %d\n", ch_0_out_0);
        // fflush(stdout);

        out_mult++;
        out_shift++;
        bias++;

        row_count--;
    }

    /* return the new output pointer with offset */
    return out_0;
}