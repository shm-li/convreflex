#!/bin/bash

OUTPUT_TO=_profile_outputs

# Total: 32
START_INPUT=320
MAX_INPUT=351

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

setup_code_softlink code_for_profiling
mkdir -p ${OUTPUT_TO}
make clean
make RUN_PROFILING=1 -j 9
make -j 9 >> temp_make_log 2>&1
if [ $? -ne 0 ]; then
    echo "make FAIL when compiling; check temp_make_log"
    exit -1
fi
for i in $(seq ${START_INPUT} ${MAX_INPUT})
do
    # set the input number in source files
    # No need to make clean; it will be too slow to start from scratch
    set_input_file_no $i
    rm test_nn
    rm obj/test_nn.o
    make RUN_PROFILING=1 -j 2
    ./test_nn > ${OUTPUT_TO}/out_${i} 2>&1
done

rm temp_make_log