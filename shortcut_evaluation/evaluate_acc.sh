#!/bin/bash

TARGET_EVAL_CODE_FOLDER=$1

SHORTCUT_OUTPUT_TO=_acc_eval_${TARGET_EVAL_CODE_FOLDER}_outputs
BASELINE_OUTPUT_TO=_acc_eval_baseline_outputs

# Total: 320
START_INPUT=0
MAX_INPUT=319


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
make RUN_BASELINE=1 -j 9 >> temp_make_log 2>&1
if [ $? -ne 0 ]; then
    echo "make FAIL when compiling the baseline version; check temp_make_log"
    exit -1
fi
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    if [ -f ${BASELINE_OUTPUT_TO}/out_${i} ]; then
        continue
    fi
    # set the input number in source files
    # No need to make clean; it will be too slow to start from scratch
    set_input_file_no $i
    rm test_nn
    rm obj/test_nn.o
    make RUN_BASELINE=1 -j 2
    ./test_nn > ${BASELINE_OUTPUT_TO}/out_${i} 2>&1
done

# If not existing, run shortcut-enabled version and get prediction results
setup_code_softlink ${TARGET_EVAL_CODE_FOLDER}
mkdir -p ${SHORTCUT_OUTPUT_TO}
make clean > temp_make_log 2>&1
make -j 9 >> temp_make_log 2>&1
if [ $? -ne 0 ]; then
    echo "make FAIL when compiling the shortcut-enabled version; check temp_make_log"
    exit -1
fi
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    if [ -f ${SHORTCUT_OUTPUT_TO}/out_${i} ]; then
        continue
    fi
    set_input_file_no $i
    rm test_nn
    rm obj/test_nn.o
    make -j 2
    ./test_nn > ${SHORTCUT_OUTPUT_TO}/out_${i} 2>&1
done

# Calculate and compare acc
# TODO: maybe it's simpler with a python script??

# By default, this is what the test_nn.c generates
reg_macro="^Label: ([0-9]+), Max: ([0-9]+).*"
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    while IFS= read -r line; do
        if [[ "$line" =~ $reg_macro ]]
        then
            label=${BASH_REMATCH[1]}
            max=${BASH_REMATCH[2]}
            if [[ "$label" == "$max" ]]
            then
                hit=$((hit + 1))
            fi
        fi
    done < "${BASELINE_OUTPUT_TO}/out_${i}"
done
cnt=$((${MAX_INPUT}-${START_INPUT}+1))
acc=$(bc <<< "scale=4; ${hit}/${cnt}*100")
result="Baseline acc: ${hit}/${cnt}=${acc}%"

sc_hit=0
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    while IFS= read -r line; do
        if [[ "$line" =~ $reg_macro ]]
        then
            label=${BASH_REMATCH[1]}
            max=${BASH_REMATCH[2]}
            if [[ "$label" == "$max" ]]
            then
                sc_hit=$((sc_hit + 1))
            fi
        fi
    done < "${SHORTCUT_OUTPUT_TO}/out_${i}"
done

sc_acc=$(bc <<< "scale=4; ${sc_hit}/${cnt}*100")
acc_diff=$(bc <<< "scale=4; ${sc_acc}-${acc}")
result="${result}, ${TARGET_EVAL_CODE_FOLDER} acc: ${sc_hit}/${cnt}=${sc_acc}%, diff: ${acc_diff}%"
echo "${result}"

rm temp_make_log