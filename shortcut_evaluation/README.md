# Example
Make sure you have run the example in [shortcut_creation](../shortcut_creation).

You should run the shortcut evaluation process under a NN's working folder. For example, ST_HandPosture CNN (HPR, hand posture recognition):
```bash
bash ../../shortcut_evaluation/evaluate_acc.sh code_conf_1200
```

The parameter is the folder that contains the model (C code) you want to evaluate. This step will run the baseline and the shortcut-enabled model with 320 inputs, of which the program outputs are stored in ```_acc_eval_baseline_outputs``` and ```_acc_eval_code_conf_1200_outputs```, respectively (if the files exist, this step is skipped). 
The code analyze the program outputs and computes accuracy difference. You should see an output at the end indicating that the shortcut-enabled model's accuracy drops by 0.0093: 
```
Baseline acc: 315/320=98.4300%, code_conf_1200 acc: 312/320=97.5000%, diff: -.9300%
```