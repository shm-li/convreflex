# Example
Make sure you have run the example in [profiling](../profiling).

You should run the shortcut creation process under a NN's working folder. For example, ST_HandPosture CNN (HPR, hand posture recognition):
```bash
python ../../shortcut_creation/ProfilingStatsParser.py gen_shortcuts 1200
```

The first parameter ```gen_shortcuts```tells the script to run at shortcut creation mode (another mode displays the profiled data). 
The parameter ```1200``` controls the shortcut selection. It is the confidence that the shortcuts make correct predictions about value clamping, estimated from the profiled data. For example, passing ```1000``` means ConvReflex will search for such shortcut triggering thresholds, that they ensures 100.0% correct clamping prediction in the profiled data. 

Here, *conf* is set to ```1200```; this is a special usage of this parameter. ConvReflex creates shortcuts with stricter thresholds than ```1000```, such that only the first 100/120 proportion of the correct clamping predictions in the profiled data are treated as "actually correct", while the rest 20/120 are ignored. In other words, *1/6* of all the edge cases are not trusted. 

The above command should create a .pkl file, ```_shortcuts_CNN2D_ST_HandPosture_8classes_quantized_ST_VL53L8CX_handposture_dataset_conf_1200.pkl```. There is also a file ```_profile_outputs_processed_cache.pkl```, which is the cached intermediate processing results from processing the profiled data, so that next time you run the shortcut creation step (even with a different *conf*), you don't have to scan all the files again. 

Now, codegen can be run again, while taking both the .tflite model and the shortcut config file as inputs:
```bash
python ../../shortcut_creation/gen_c_code_w_shortcut.py CNN2D_ST_HandPosture_8classes_quantized.tflite _shortcuts_CNN2D_ST_HandPosture_8classes_quantized_ST_VL53L8CX_handposture_dataset_conf_1200.pkl code_conf_1200
```

Three parameters are given: the .tflite file path, the shortcut config .pkl file path, and an output folder name. The folder ```code_conf_1200``` should be created after this step, which contains the generated code. 