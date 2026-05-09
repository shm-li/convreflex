from code_generator.CodegenUtilTFlite import GenerateSourceFilesFromTFlite
import os
import shutil
import sys

if __name__ == "__main__":
    tflite_path = sys.argv[1]
    termination_check_pos_file = sys.argv[2] 
    target_codegen_name = sys.argv[3]
    print("WARNING: Also sending in pre-generated termination check position in file {:s}", termination_check_pos_file)

    if os.path.exists("./codegen"):
        shutil.rmtree("./codegen")

    _ = GenerateSourceFilesFromTFlite(
        tflite_path,
        life_cycle_path="./lifecycle.png",
        redundancy_omitting_mode="clamping_predicting",
        termination_check_file=termination_check_pos_file
    )

    shutil.move("./codegen", "./{:s}".format(target_codegen_name))
    os.remove("./lifecycle.png")