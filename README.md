## Prerequisites

Before building this project, ensure you have the following installed:

1. **CMake** (version 3.22 or higher)
   ```bash
   brew install cmake
   ```

2. **JUCE Framework**
   - Install JUCE in `~/Documents/JUCE`
   - Download from: https://juce.com/get-juce/
   - **Note**: This path is not mandatory and can be adjusted by modifying the `JUCE_DIR` variable in `rtautolume/CMakeLists.txt`

3. **LibTorch** (PyTorch C++ Library)
   - **Required**: install in `./rtautolume/plugin/libs/libtorch`
   - Download the latest version from: https://pytorch.org/get-started/locally/
   - For macOS, select:
     - PyTorch Build: Stable
     - Your OS: Mac
     - Package: LibTorch
     - Language: C++/Java
     - Compute Platform: Default
   - **Note**: This path is not mandatory and can be adjusted by modifying the `CMAKE_PREFIX_PATH` in `rtautolume/plugin/CMakeLists.txt`

4. **C++ Compiler** with C++20 support
   - Xcode Command Line Tools (macOS)
   ```bash
   xcode-select --install
   ```

## Building the Project (Xcode)

1. **Navigate to the rtautolume directory:**
   ```bash
   cd rtautolume
   ```

2. **Create a build directory:**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure the Xcode project with CMake:**
   ```bash
   cmake -G Xcode ..
   ```

## Python Inference Script

The `./autolume` directory contains a Python script for running inference with TorchScript models.

### Setup

1. **Navigate to the autolume directory:**
   ```bash
   cd autolume
   ```

2. **Create a virtual environment with Python 3.10:**
   ```bash
   python3.10 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

For comprehensive help on how to run the inference script:
```bash
python torchscript_inference.py -h
```

This will display all available options and usage instructions.

## Model files
You can download the model file `stylegan2_notrain.pt` and watch the demo from this [shared folder](https://drive.google.com/drive/folders/1xToVOaCljGrI8JMnP6MJeBwr3ftp-8mV?)

## License

This codebase is subject to the [Creative Commons License](LICENSE) and [NVIDIA License](./autolume/LICENSE_NVIDIA.txt), as well as any licensing terms imposed by the original developers of autolume at [https://github.com/Metacreation-Lab/autolume](https://github.com/Metacreation-Lab/autolume).
