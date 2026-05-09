from code_generator.CodegenUtilTFlite import GenerateSourceFilesFromTFlite
import os
import shutil
import sys

if __name__ == "__main__":
    tflite_path = sys.argv[1]

    if os.path.exists("./codegen"):
        shutil.rmtree("./codegen")

    _ = GenerateSourceFilesFromTFlite(
        tflite_path,
        life_cycle_path="./lifecycle.png",
        redundancy_omitting_mode="clamping_predicting",
        termination_check_file=None
    )

    code_folder = "code_for_profiling"
    shutil.move("./codegen", "./{:s}".format(code_folder))
    os.remove("./lifecycle.png")