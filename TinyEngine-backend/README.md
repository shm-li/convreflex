# TinyEngine backend for Conv Reflex
This is implemented based on a modified version of TinyEngine: [TinyEngine without dependency on arm-v7e insts](https://github.com/shm-li/tinyengine-armv6m). Supports for ConvReflex code generation are added. If you want to check TinyEngine's original README, see [OLD_README.md](OLD_README.md).

## Before using this
Make sure you have installed the requirements (better use a Python virtual environment)! Below is a list of package versions that I use on my MacOS (Python 3.10.13): 
|name | version |
|-----|---------|
|flatbuffers|25.2.10|
|matplotlib|3.10.5|
|numpy|2.2.6|
|tqdm|4.67.1|

Then, run the below command to config the Python paths (under this directory), so that the packages can be found when you run TinyEngine for code generation. 
```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```