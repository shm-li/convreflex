# Example
You should run profiling under a NN's working folder. For example, ST_HandPosture CNN (HPR, hand posture recognition):
```bash
cd convreflex/nns_working_folder/CNN2D_ST_HandPosture_8classes_quantized_ST_VL53L8CX_handposture_dataset
```

Generate the C code from the .tflite file (already in the folder):
```bash
python ../../profiling/gen_c_code.py CNN2D_ST_HandPosture_8classes_quantized.tflite
```

This should create a folder ```code_for_profiling```. 

Then, run the model with sample inputs:
```bash
bash ../../profiling/profile_sample_inputs.sh
```

This will create a folder ```_profile_outputs``` which holds all profiling data generated from each input. 
