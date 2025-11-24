/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   add.c
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


#include <math.h>
#include <stdio.h>
#include "arm_math.h"
#include "tinyengine_function.h"

inline int32_t Add(int32_t a, int32_t b) {
  return a + b;
}
inline int32_t ShiftRight(int32_t a, int offset) {
  return a >> offset;
}
inline int32_t BitAnd(int32_t a, int32_t b) {
  return a & b;
}
inline int32_t BitNot(int32_t a) {
  return ~a;
}
inline int32_t MaskIfNonZero(int32_t a) {
  static const int32_t zero = 0;
  return a ? BitNot(zero) : zero;
}
inline int32_t MaskIfGreaterThan(int32_t a, int32_t b) {
  return MaskIfNonZero(a > b);
}
inline int32_t MaskIfLessThan(int32_t a, int32_t b) {
  return MaskIfNonZero(a < b);
}

static inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  int64_t a_64 = a;
  int64_t b_64 = b;
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (1ll << 31));
  return a == b && a == -2147483648 ? 2147483647 : ab_x2_high32;
}

static inline  int32_t RoundingDivideByPOT(int32_t x, int exponent) {
  const int32_t mask = ((1ll << exponent) - 1);
  const int32_t zero = (0);
  const int32_t one = (1);
  const int32_t remainder = BitAnd(x, mask);
  const int32_t threshold = Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
  return Add(ShiftRight(x, exponent), BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}

static inline int32_t MultiplyByQuantizedMultiplier(
		int32_t x, int32_t quantized_multiplier, int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}

tinyengine_status mul_broadcast_axis_1_2(
      const int16_t input_h, const int16_t input_w, const int16_t input_c, 
      const int32_t input1_offset, const int32_t input2_offset,
      const int32_t output_offset,
      const int32_t output_multiplier, const int32_t output_shift,
      const int8_t* input1_data, const int8_t* input2_data, 
			const int32_t output_activation_min, const int32_t output_activation_max,
      int8_t* output_data) {
  for (int h = 0; h < input_h; ++h) {
    for (int w = 0; w < input_w; ++w) {
      for (int c = 0; c < input_c; ++c) {
        const int32_t input1_val = input1_offset + input1_data[(h * input_w + w) * input_c + c];
        const int32_t input2_val = input2_offset + input2_data[c];
        // printf("MUL: %d (%d + %d) * %d (%d + %d)\r\n", input1_val, input1_offset, input1_data[(h * input_w + w) * input_c + c], input2_val, input2_offset, input2_data[c]);
        // fflush(stdout);
        const int32_t raw_output =
            MultiplyByQuantizedMultiplier(
                input1_val * input2_val, output_multiplier, output_shift) \
                  + output_offset;
        
        const int32_t clamped_output = TN_MIN(output_activation_max,
            TN_MAX(output_activation_min, raw_output));
        output_data[(h * input_w + w) * input_c + c] = clamped_output;
      }
    }
  }
  // for (int i = 0; i < size; ++i) {
  //   printf("%d: %d\n", i, output_data[i]);
  // }
  // printf("end printing outputs\n");
}
