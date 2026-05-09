# ConvReflex: Efficient Ultra-Low-Power CNN Inference via Clamping Prediction

ConvReflex is presented at SenSys '26: [https://doi.org/10.1145/3774906.3802781](https://doi.org/10.1145/3774906.3802781).

## What is it?

This toolchain is designed to work in the tflite-to-C code generation phase(codegen implemented based on a modified version of TinyEngine: [TinyEngine without dependency on arm-v7e insts](https://github.com/shm-li/tinyengine-armv6m)). ConvReflex builds "shortcuts" in the generated convolution kernel code; these shortcuts allow kernels to predict whether the final conv result is **clamped** to -128 (lower bound of INT8), and if so, ignore the remaining computations and jump to the result. 

## How does it work?

**Profiling:** ConvReflex starts by profiling an NN model's behavior, using some sample input data--what we use is a part of the test set. 

**Shortcut creation:** By analyzing the profiled data, ConvReflex decides the exact shortcut configurations of for each convolution kernel (some may have none). Each shortcut is defined by the computation step that it is inserted into (e.g. after 10/18 MACs are finished), and the threshold value to trigger the shortcut (e.g. intermediate result < -4200). 

**Shortcut evaluation:** ConvReflex can generate C code containing the shortcuts. The accuracy of the shortcut-enabled model can be evaluated against the baseline version. 

**Adjusting:** The shortcut creation step has an adjustable parameter. We call it *conf*: it is the confidence that the shortcuts predict value clamping correctly, estimated from profiled data. This can be adjusted, and the shortcut creation and evaluation steps can be run again (they will take less time than you first run them!), until a model shows acceptable accuracy loss. 