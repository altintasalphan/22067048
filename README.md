# MKT3434_2025: Machine Learning Integration Framework

[![Python 3.9-3.12](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU%20Accelerated-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository provides a comprehensive GUI framework for developing and integrating machine learning methods, created for the MKT3434 Course at Yildiz Technical University's Department of Mechatronics Engineering. The application leverages PyQt6 to deliver an intuitive interface that supports both classical machine learning techniques and advanced deep learning implementations, now enhanced with dimensionality reduction and cross-validation features critical for real-world robotics and mechatronics applications.

**Course Instructor:** [Asst. Prof. Dr. Ertuğrul Bayraktar](https://github.com/bayraktare)  
**Developed by:** [Alphan Bartu ALTINTAŞ](https://github.com/altintasalphan) - Mechatronics Engineering Student (22067048)

## 🔍 Key Features

- **Intuitive GUI Interface**: Built with PyQt6 for cross-platform compatibility
- **Comprehensive ML Support**: Integration with popular machine learning libraries
- **GPU Acceleration**: CUDA support for enhanced performance on compatible hardware
- **Extensible Architecture**: Designed to accommodate additional functionalities
- **Advanced Loss Function Selection**: Customize models with different loss functions
- **Missing Data Handling**: Multiple strategies for handling missing values
- **Bayesian Methods**: Implementations of Bayesian algorithms with customizable parameters
- **Support Vector Machine**: Classification and regression support with multiple kernels
- **Visualization Capabilities**: Enhanced with Plotly for interactive 2D/3D plots
- **Dimensionality Reduction**: PCA, LDA, t-SNE, UMAP, and K-Means with advanced features
- **Cross-Validation**: K-fold cross-validation with user-selectable k and detailed metrics

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
  - See the [list of CUDA®-enabled GPU cards](https://developer.nvidia.com/cuda-gpus)
- **For CPU Setup**: Intel (x64) or Arm64 CPU recommended (AMD CPU compatibility not guaranteed)
  - **Note:** TensorFlow binaries use AVX instructions which may not run on older CPUs
  - **Note:** Starting with TensorFlow 2.10, Windows CPU-builds for x86/x64 processors are built, maintained, tested, and released by Intel through the `tensorflow-intel` package

#### Software Prerequisites
- Python 3.9-3.12
- pip version 19.0+ for Linux/Windows, 20.3+ for macOS
- Windows Native: Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019
- NVIDIA® GPU drivers:
  - Linux: >= 525.60.13
  - WSL on Windows: >= 528.33

### Installation Guide

#### 1. Create a Virtual Environment (Recommended)
```bash
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate  # Linux/macOS
# OR
\path\to\new\virtual\environment\Scripts\activate  # Windows
```

#### 2. Install TensorFlow

**For GPU Support:**
```bash
python -m pip install --upgrade pip
python -m pip install tensorflow[and-cuda]
```

**For CPU-only:**
```bash
python -m pip install --upgrade pip
python -m pip install tensorflow  # or pip install tensorflow-cpu
```

**Note:** TensorFlow versions above 2.10 do not support GPU on Windows Native.

#### 3. Verify Installation

**For GPU Configuration:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If GPUs are listed, your GPU setup is working correctly.

**For CPU Configuration:**
```bash
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
If a tensor is returned, you've installed TensorFlow successfully.

#### 4. Install Additional Dependencies
```bash
python -m pip install numpy pandas matplotlib PyQt6 scikit-learn torch torchvision torchaudio opencv-python opencv-contrib-python scipy fastai kornia plotly umap-learn
```

## 📊 Using the Enhanced Features

### Dimensionality Reduction
1. Navigate to the **Dimensionality Reduction** tab
2. Configure parameters for:
   - **PCA**: Select number of components and visualize explained variance
   - **LDA**: Supervised reduction with class separation metrics (e.g., silhouette score)
   - **t-SNE**: Interactive 2D/3D projections with perplexity tuning
   - **UMAP**: Faster alternative to t-SNE with configurable neighbors and distance
   - **K-Means**: Configurable clusters with elbow method support
3. Click "Train" for individual methods or "Compare Dimensionality Reduction Methods" to visualize multiple methods side by side

### Cross-Validation
1. In the **Data Management** section, select "K-Fold Cross-Validation" under "Validation Method"
2. Set the number of folds (k) using the spin box (2-10)
3. Train a model under the **Classical ML** tab to see cross-validation metrics (mean and std of accuracy or MSE)

### Interactive Visualization
- Uses Plotly for interactive 2D/3D scatter plots in dimensionality reduction
- Elbow method plots and method comparisons are displayed in separate dialogs

## 🖥️ Development Environment

This GUI framework has been developed and tested in the following environment:

- **Operating System**: Windows 11 IoT Enterprise LTSC with WSL2 (Ubuntu 24.04.02 LTS)
- **Python**: 3.12.3 within a Virtual Environment (venv)
- **Hardware**:
  - **CPU**: Intel i7-11370H
  - **GPU**: NVIDIA RTX 3050 Ti with Studio Driver 576.02
- **ML Framework Configuration**:
  - CUDA 12.9
  - cuDNN 9.10.0
  - TensorRT 10.11
- **IDE**: Visual Studio Code

## ⚠️ Known Issues and Solutions

When running with GPU setup on WSL2, you may encounter Qt/XCB plugin loading errors. To resolve this:

**Temporary Solution**:
```bash
export QT_QPA_PLATFORM=xcb
python 22067048.py
```

**Permanent Solution** (requires virtual environment):

1. Add to virtual environment activation script:
   ```bash
   # Save original QT_QPA_PLATFORM if it exists
   if [ -n "${QT_QPA_PLATFORM:-}" ] ; then
       _OLD_QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-}"
   fi

   # Set Qt to use xcb
   export QT_QPA_PLATFORM=xcb
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

## 📚 Additional Resources

For further assistance with installation issues or for more detailed information about the dependencies, please refer to these official resources:

- **TensorFlow Installation**: [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip)
- **CUDA Installation**: [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- **cuDNN Installation**: [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/)
- **TensorRT Installation** (optional): [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- **CUDA-Enabled GPUs**: [List of CUDA-Enabled NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)
- **Plotly**: [Plotly Python Documentation](https://plotly.com/python/)
- **UMAP**: [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/)

If you encounter any problems with running the program or installing prerequisites, the resources above should be your first reference point for troubleshooting.

## 📞 Support

For issues related to this implementation, please open an issue in this repository. For questions about the original framework, please refer to the [original repository](https://github.com/bayraktare/MKT3434_2025).

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
