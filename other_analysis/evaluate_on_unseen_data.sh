#!bin/bash

TARGET_EVAL_CODE_FOLDER=$1

SHORTCUT_OUTPUT_TO=_deployment_eval_${TARGET_EVAL_CODE_FOLDER}_outputs
BASELINE_OUTPUT_TO=_deployment_eval_baseline_outputs

# Total: 32
# Start from 325 so only unseen data is used
START_INPUT=352
MAX_INPUT=383

# Error handling
if [ -z "${TARGET_EVAL_CODE_FOLDER}" ]; then
    echo "Provide the codegen folder you want to evaluate acc on"
    exit 1;
fi

setup_code_softlink () {
    rm Source
    rm Include
    ln -s ${1}/Source Source 
    ln -s ${1}/Include Include
}

set_input_file_no () {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "{s/#include \"inputs\/input_[[:digit:]]*\.h\"/#include \"inputs\/input_${1}.h\"/;}" test_setup_info.h
        sed -i '' "{s/#define INPUT_NUM [[:digit:]]*/#define INPUT_NUM ${1}/;}" test_setup_info.h
    else # For Linux
        sed -i "{s/#include \"inputs\/input_[[:digit:]]*\.h\"/#include \"inputs\/input_${1}.h\"/;}" test_setup_info.h
        sed -i "{s/#define INPUT_NUM [[:digit:]]*/#define INPUT_NUM ${1}/;}" test_setup_info.h
    fi
}

# If not existing, run baseline and get prediction results
setup_code_softlink code_for_profiling # should be okay to use any codegen
mkdir -p ${BASELINE_OUTPUT_TO}
make clean > temp_make_log 2>&1
make RUN_PROFILING=1 -j 9 >> temp_make_log 2>&1
if [ $? -ne 0 ]; then
    echo "make FAIL when compiling the baseline version; check temp_make_log"
    exit -1
fi
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    if [ -f ${BASELINE_OUTPUT_TO}/out_${i} ]; then
        echo "Profiling baseline model: Skipping input number ${i} since ouptut file exists"
        continue
    fi
    # set the input number in source files
    # No need to make clean; it will be too slow to start from scratch
    set_input_file_no $i
    rm test_nn
    rm obj/test_nn.o
    make RUN_PROFILING=1 -j 2
    ./test_nn > ${BASELINE_OUTPUT_TO}/out_${i} 2>&1
done

# If not existing, run shortcut-enabled code and get prediction results
setup_code_softlink ${TARGET_EVAL_CODE_FOLDER} # should be okay to use any codegen
mkdir -p ${SHORTCUT_OUTPUT_TO}
make clean > temp_make_log 2>&1
make RUN_PROFILING=1 -j 9 >> temp_make_log 2>&1
if [ $? -ne 0 ]; then
    echo "make FAIL when compiling the baseline version; check temp_make_log"
    exit -1
fi
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    if [ -f ${SHORTCUT_OUTPUT_TO}/out_${i} ]; then
        echo "Profiling shortcut-enabled model: skipping input number ${i} since ouptut file exists"
        continue
    fi
    # set the input number in source files
    # No need to make clean; it will be too slow to start from scratch
    set_input_file_no $i
    rm test_nn
    rm obj/test_nn.o
    make RUN_PROFILING=1 -j 2
    ./test_nn > ${SHORTCUT_OUTPUT_TO}/out_${i} 2>&1
done

# Get number of MACs that are skipped from shortcut-enabled profiling
python ../../shortcut_creation/ProfilingStatsParser.py display ${SHORTCUT_OUTPUT_TO} > _deployment_eval_${TARGET_EVAL_CODE_FOLDER}_analyze
reg_macro="^Total: terminated ([0-9]+) / .*"
while IFS= read -r line; do
    if [[ "$line" =~ $reg_macro ]]
    then
        skipped_steps=${BASH_REMATCH[1]}
    fi
done < "_deployment_eval_${TARGET_EVAL_CODE_FOLDER}_analyze"
# Get number of total steps that are executed in conv, depthwise-conv and 
#   fully connected layers (these take up almost all the comp time!)
# Note: We can't use the total step number in the shortcut-enabled model's
#   profiled data. It is possible that in that code, not all these layers 
#   have shortcuts. 
python ../../shortcut_creation/ProfilingStatsParser.py display ${BASELINE_OUTPUT_TO} > _deployment_eval_baseline_analyze
reg_macro="^Total: terminated [0-9+] / ([0-9]+).*"
while IFS= read -r line; do
    if [[ "$line" =~ $reg_macro ]]
    then
        total_steps=${BASH_REMATCH[1]}
    fi
done < "_deployment_eval_baseline_analyze"

skipped_pct=$(bc <<< "scale=4; ${skipped_steps}/${total_steps}*100")
echo "Pct. of computations skipped in ${TARGET_EVAL_CODE_FOLDER}, compared to not having shortcuts: ${skipped_pct}%"

# Or keep them if you like
rm _deployment_eval_${TARGET_EVAL_CODE_FOLDER}_analyze
rm _deployment_eval_baseline_analyze