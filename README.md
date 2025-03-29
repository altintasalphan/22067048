# MKT3434_2025: Machine Learning Integration Framework

[![Python 3.9-3.12](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU%20Accelerated-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository provides a comprehensive GUI framework for developing and integrating machine learning methods, created for the MKT3434 Course at Yildiz Technical University's Department of Mechatronics Engineering. The application leverages PyQt6 to deliver an intuitive interface that supports both classical machine learning techniques and advanced deep learning implementations.

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
- **Visualization Capabilities**: Comprehensive visualization of model performance and data

## 🧩 Enhanced Features and Improvements

### Classical Machine Learning
- **Support Vector Machine (SVM)**: Both classification (SVC) and regression (SVR) implementations
- **Kernel Selection**: Linear, RBF, Polynomial, and Sigmoid kernels
- **SGD Classification/Regression**: Stochastic Gradient Descent with different loss functions
- **Hyperparameter Tuning**: Fine-grained control over C, epsilon (for SVR), and kernel parameters

### Loss Function Customization
- **Regression Loss Functions**: MSE, MAE, Huber Loss
- **Classification Loss Functions**: Cross-Entropy, Hinge Loss
- **Dynamic Model Training**: Updates training logic based on selected loss function

### Bayesian Methods
- **Gaussian Naive Bayes**: Configurable var_smoothing parameter
- **Prior Probabilities**: Choose between uniform or custom prior probabilities
- **Visualization**: Dedicated visualization for Bayesian model performance

### Missing Data Handling
- **Multiple Imputation Methods**: Mean, Median, Most Frequent, Forward Fill, Backward Fill
- **Testing Framework**: Option to introduce missing values for testing imputation strategies
- **Method Comparison**: Tool to compare the effectiveness of different imputation methods
- **Integration with scikit-learn**: Uses SimpleImputer for robust missing data handling

### Deep Learning Enhancements
- **Layer Configuration**: Improved interface for adding and configuring neural network layers
- **Loss Function Selection**: Apply different loss functions to neural networks
- **Model Visualization**: Enhanced visualization of training history and model performance
- **CNN Support**: Convolutional Neural Network architecture builder

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
python -m pip install numpy pandas matplotlib PyQt6 scikit-learn torch torchvision torchaudio opencv-python opencv-contrib-python scipy fastai kornia scikit-learn>=1.0.0
```

## 📊 Using the Enhanced Features

### Missing Data Handling
1. Load a dataset and check "Introduce Missing Values" to test imputation methods
2. Select an imputation method from the dropdown menu
3. Use the "Compare Imputation Methods" button to evaluate different strategies

### Loss Function Selection
1. Navigate to the Classical ML or Deep Learning tab
2. Select the desired loss function from the radio buttons
3. Train your model with the selected loss function

### Support Vector Machine/Regression
1. Go to the Classical ML tab
2. Choose from SVM (classification) or SVR (regression)
3. Configure kernel type and parameters
4. Click "Train" to build and evaluate the model

### Bayesian Methods
1. Navigate to the Bayesian Methods tab
2. Configure var_smoothing parameter
3. Choose between uniform or custom prior probabilities
4. Click "Train Naive Bayes" to build the model

## ⚠️ Known Issues and Solutions

When running with GPU setup on WSL2, you may encounter Qt/XCB plugin loading errors. To resolve this:

**Temporary Solution**:
```bash
export QT_QPA_PLATFORM=offscreen
python 22067048.py
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
