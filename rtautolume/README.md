## Motivation
To reduce reliance on NVIDIA licensing, and not a fan of Python-based UI

## Prerequisites

Before building this project, ensure you have the following installed:

1. **CMake** (version 3.22 or higher)

2. **JUCE Framework**
   - Download from: https://juce.com/get-juce/
   - Install JUCE in `~/Documents/JUCE`
     - **Note**: This path is not mandatory and can be adjusted by modifying the `JUCE_DIR` variable in `rtautolume/CMakeLists.txt`
  
3. **LibTorch** (PyTorch C++ Library)
   - Download the latest version from: https://pytorch.org/get-started/locally/
   - Install in `./rtautolume/plugin/libs/libtorch`
     - **Note**: This path is not mandatory and can be adjusted by modifying the `CMAKE_PREFIX_PATH` in `rtautolume/plugin/CMakeLists.txt`

4. **C++ Compiler** with C++20 support

## Building the Project
**MacOS**
   ```bash
   mkdir build
   cd build
   cmake -G Xcode ..
   ```

**Windows**
   ```bash
   mkdir build
   cd build
   cmake -G "Your Visual Studio Version (eg. Visual Studio 17 2022)" ..
   ```

**Linux**
   ```bash
   cmake -B build
   cd build
   make -j${nproc}
   ```

## Loading a pretrained model
For now, use [https://drive.google.com/file/d/1PITDmC624wk1FWK7WmRMgM5nC2NUjmd8/view?usp=sharing](https://drive.google.com/file/d/1PITDmC624wk1FWK7WmRMgM5nC2NUjmd8/view?usp=sharing)

## Non-exhaustive TODO list
Need to implement an audio resampler at 16khz before feeding the audio input to feature extraction