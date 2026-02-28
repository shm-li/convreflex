
#include "tinyengine_function.h"

tinyengine_status quantize_float32_to_int8(
		const char *inputptr, const uint16_t size,
		const float scale, const int32_t zero_point,
		const int32_t min_val, const int32_t max_val,
		int8_t *output)
{
	float* input = (float*)inputptr;
	for (int i = 0; i < size; i++) {
		const float val = input[i];
		int32_t res = roundf(val / scale);
		res += zero_point;

		res = ((res < min_val) ? min_val : res);
		res = ((res > max_val) ? max_val : res);
		output[i] = (int8_t)res;
	}
}
