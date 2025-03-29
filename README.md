# MKT3434_2025: Machine Learning Integration Framework

[![Python 3.9-3.12](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU%20Accelerated-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository provides a comprehensive GUI framework for developing and integrating machine learning methods, created for the MKT3434 Course at Yildiz Technical University's Department of Mechatronics Engineering. The application leverages PyQt6 to deliver an intuitive interface that supports both classical machine learning techniques and advanced deep learning implementations.

**Course Instructor:** [Dr. EB](https://github.com/bayraktare)  
**Developed by:** [Alphan Bartu ALTINTAŞ](https://github.com/altintasalphan) - Mechatronics Engineering Student (22067048)

## 🔍 Key Features

- **Intuitive GUI Interface**: Built with PyQt6 for cross-platform compatibility
- **Comprehensive ML Support**: Integration with popular machine learning libraries
- **GPU Acceleration**: CUDA support for enhanced performance on compatible hardware
- **Extensible Architecture**: Designed to accommodate additional functionalities

## 🤝 Repository Information

This project is a fork developed as part of the Yildiz Technical University MKT3434 Course (2025). For the base GUI and original implementation, please refer to the [original repository](https://github.com/bayraktare/MKT3434_2025).

## 🚀 Getting Started

### System Requirements

#### Operating Systems
- **Ubuntu**: 16.04+ (64-bit)
- **Windows**: Windows 7+ (64-bit) or Windows 10 19044+ with WSL2
- **macOS**: 12.0 Monterey or higher (64-bit) (Note: no GPU support)

#### Hardware Requirements
- **For GPU Setup**: NVIDIA® CUDA®-enabled GPU
- **For CPU Setup**: Intel (x64) or Arm64 CPU recommended (AMD CPU compatibility not guaranteed)

#### Software Prerequisites
- Python 3.9-3.12
- pip version 19.0+ for Linux/Windows, 20.3+ for macOS
- Windows Native: Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019
- NVIDIA® GPU drivers:
  - Linux: >= 525.60.13
  - WSL on Windows: >= 528.33

### Installation Guide

1. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv /path/to/new/virtual/environment
   source /path/to/new/virtual/environment/bin/activate  # Linux/macOS
   # OR
   \path\to\new\virtual\environment\Scripts\activate  # Windows
   ```

2. **Install TensorFlow with GPU Support**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install tensorflow[and-cuda]
   ```

3. **Verify GPU Configuration**
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   If GPUs are listed, your GPU setup is working correctly.

4. **Install Additional Dependencies**
   ```bash
   python -m pip install numpy pandas matplotlib PyQt6 scikit-learn torch torchvision torchaudio opencv-python opencv-contrib-python scipy fastai kornia
   ```

## 🖥️ Development Environment

This GUI framework has been developed and tested in the following environment:

- **Operating System**: WSL2 (Ubuntu 24.04.01 LTS) on Windows 11 IoT Enterprise LTSC (24H2 Build 26100.3476)
- **Python**: 3.12.3 within a Virtual Environment (venv)
- **GPU Configuration**:
  - CUDA 12.8
  - TensorRT 10.9
  - cuDNN 9.8.0
  - NVIDIA RTX 3050 Ti with Studio Driver 572.83
- **CPU**: Intel i7-11370H
- **IDE**: Visual Studio Code

## ⚠️ Known Issues and Solutions

When running with GPU setup on WSL2, you may encounter Qt/XCB plugin loading errors. To resolve this:

**Temporary Solution**:
```bash
export QT_QPA_PLATFORM=offscreen
python base_gui_for_MKT3434_by_eb.py
```

**Permanent Solution** (requires virtual environment):

1. Add to virtual environment activation script:
   ```bash
   # Save original QT_QPA_PLATFORM if it exists
   if [ -n "${QT_QPA_PLATFORM:-}" ] ; then
       _OLD_QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-}"
   fi

   # Set Qt to use offscreen rendering
   export QT_QPA_PLATFORM=offscreen
   ```

2. Add to deactivation section in the same script:
   ```bash
   # Unset Qt platform environment variable
   if [ -n "${_OLD_QT_QPA_PLATFORM:-}" ] ; then
       QT_QPA_PLATFORM="${_OLD_QT_QPA_PLATFORM:-}"
       export QT_QPA_PLATFORM
       unset _OLD_QT_QPA_PLATFORM
   else
       unset QT_QPA_PLATFORM
   fi
   ```

## 📞 Support

For issues related to this implementation, please open an issue in this repository. For questions about the original framework, please refer to the [original repository](https://github.com/bayraktare/MKT3434_2025).

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
