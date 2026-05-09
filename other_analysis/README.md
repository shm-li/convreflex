# Example: evaluate created model on unseen data
Make sure you have at least generated a model code with shortcuts. This is done after finishing [shortcut_creation](../shortcut_creation). 

You should run the shortcut evaluation process under a NN's working folder. For example, ST_HandPosture CNN (HPR, hand posture recognition):
```bash
bash ../../other_analysis/evaluate_on_unseen_data.sh code_conf_1200
```

The parameter provides the code generation folder. This script will run the models *with profiling mode*, and generate the profiled outputs in ```_deployment_eval_baseline_outputs``` and ```_deployment_eval_code_conf_1200_outputs``` (this is skipped if these files exist). 
Then, the script analyzes the profiled data to extract the total number of MACs executed and the number of MACs that are skipped. You will see the percentage of skipped computations (31.1%) in the last line of output: 
```
Pct. of computations skipped in code_conf_1200, compared to not having shortcuts: 31.1300%
```