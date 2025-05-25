# MKT3434_2025: Advanced Machine Learning Integration Framework

[![Python 3.9-3.12](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU%20Accelerated-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.3-green.svg)](https://github.com/altintasalphan/MKT3434_2025)

## 📋 Overview

This repository provides a comprehensive GUI framework for developing and integrating advanced machine learning methods, created for the MKT3434 Course at Yildiz Technical University's Department of Mechatronics Engineering. The application leverages PyQt6 to deliver an intuitive interface that supports both classical machine learning techniques and cutting-edge deep learning implementations, now enhanced with advanced neural network design capabilities, transfer learning integration, real-time training monitoring, and generative model frameworks.

**Course Instructor:** [Asst. Prof. Dr. Ertuğrul Bayraktar](https://github.com/bayraktare)  
**Developed by:** [Alphan Bartu ALTINTAŞ](https://github.com/altintasalphan) - Mechatronics Engineering Student (22067048)

## 🔍 Key Features

### 🎯 Advanced Deep Learning Capabilities (NEW in v0.1.3)
- **Interactive Neural Network Designer**: Visual layer-by-layer architecture construction with comprehensive parameter configuration
- **Enhanced Layer Support**: Dense, Convolutional (2D/1D), Recurrent (LSTM/GRU/SimpleRNN), Pooling, Normalization, Dropout, and specialized layers
- **Real-time Training Monitoring**: Live loss/accuracy visualization with gradient flow monitoring and training log streaming
- **Professional Optimizer Suite**: Adam, SGD, RMSprop, AdaGrad, Adadelta with advanced learning rate scheduling
- **Comprehensive Regularization**: L2 regularization, Dropout controls, Early stopping with validation monitoring
- **Architecture Analysis System**: Automated architecture explanation with pattern recognition and optimization recommendations

### 🚀 Transfer Learning & Fine-tuning (NEW in v0.1.3)
- **Pre-trained Model Integration**: VGG16/19, ResNet50/101, InceptionV3, Xception, MobileNet, DenseNet121, EfficientNetB0
- **Flexible Fine-tuning Controls**: Layer freezing/unfreezing, custom classifier addition, learning rate configuration
- **Advanced Image Augmentation**: Rotation, flipping, zoom, brightness adjustment with real-time parameter control
- **Model Persistence**: Save/load fine-tuned models in HDF5 and SavedModel formats with metadata preservation

### 🎨 Generative Models Framework (NEW in v0.1.3)
- **GAN Infrastructure**: Vanilla GAN, DCGAN, Conditional GAN with configurable architectures
- **Training Pipeline**: Adversarial training loop structure with loss balancing and stability monitoring
- **Sample Generation**: Framework for real-time sample generation and quality evaluation
- **Extensible Design**: Foundation for future implementation of VAE, Transformer models, and Diffusion models

### 📊 Enhanced Data Management & Preprocessing
- **Extended Dataset Support**: MNIST, CIFAR-10, Fashion-MNIST, Custom CSV datasets with intelligent preprocessing
- **Advanced Missing Value Handling**: Multiple imputation strategies with performance comparison and recommendation system
- **Comprehensive Scaling Options**: Standard, Min-Max, Robust scaling with automatic data type detection
- **Data Quality Assessment**: Automatic missing value detection with visualization and handling recommendations

### 🧠 Intelligent Model Analysis & Optimization
- **Architecture Explanation Engine**: Automated analysis of network designs with pattern recognition and recommendations
- **Optimizer Comparison Tool**: Detailed comparison of optimization algorithms with use-case recommendations
- **Real-time Performance Monitoring**: Live training metrics with gradient monitoring and convergence analysis
- **Model Persistence System**: Save/load neural network architectures with complete configuration preservation

### 📈 Professional Visualization & Reporting
- **Multi-threaded Training**: Non-blocking training operations with responsive UI and real-time progress tracking
- **Advanced Plotting System**: Training curves, confusion matrices, PCA visualizations, gradient monitoring
- **Comprehensive Metrics Dashboard**: Detailed performance analysis with statistical summaries and recommendations
- **Export Capabilities**: Save models, architectures, and training results with complete metadata

### 🔧 Robust Classical Machine Learning Suite
- **Comprehensive Algorithm Support**: Linear/Logistic Regression, SVM, Random Forest, KNN, Naive Bayes with Bayesian parameter tuning
- **Advanced Loss Function Selection**: Customizable loss functions for classification and regression tasks
- **Cross-validation Support**: K-fold validation with detailed statistical analysis
- **Dimensionality Reduction**: PCA, K-Means clustering with visualization and quality metrics

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

## 🎯 Using the Enhanced Features

### Advanced Deep Learning Workflow
1. Navigate to the **Advanced Deep Learning** tab
2. **Design Your Architecture**:
   - Click "Add Layer" to open the comprehensive layer configuration dialog
   - Select layer types: Dense, Conv2D/1D, LSTM/GRU, BatchNormalization, Dropout, etc.
   - Configure parameters: units, activation functions, kernel sizes, regularization
   - Use the tree view to visualize your architecture in real-time
3. **Configure Advanced Training**:
   - Select optimizer (Adam, SGD, RMSprop, AdaGrad, Adadelta)
   - Set learning rate with optional scheduling (Step Decay, Exponential, Cosine Annealing)
   - Configure regularization: L2 regularization, Early stopping with patience control
   - Enable gradient monitoring for training analysis
4. **Monitor Training in Real-time**:
   - Watch live training logs with timestamped events
   - Monitor gradient flow with histogram visualization
   - Track training progress with responsive progress bars
5. **Save and Analyze Results**:
   - Save complete model architectures in JSON format
   - Export trained models in HDF5 or SavedModel format
   - Use "Explain Architecture" for automated analysis and recommendations

### Transfer Learning Workflow
1. Go to the **Transfer Learning** tab
2. **Select Pre-trained Model**:
   - Choose from VGG16/19, ResNet50/101, InceptionV3, Xception, MobileNet, DenseNet121, EfficientNetB0
   - Click "Load Pre-trained Model" to download and initialize (first time only)
   - Review model information: parameters, layers, input/output shapes
3. **Configure Fine-tuning Strategy**:
   - Choose to freeze base layers or unfreeze top N layers
   - Set fine-tuning learning rate (typically 10-100x lower than base training)
   - Enable custom classifier addition for domain-specific tasks
4. **Set Up Data Augmentation**:
   - Enable rotation, horizontal flip, zoom, brightness adjustment
   - Configure augmentation intensity for optimal generalization
5. **Start Transfer Learning**:
   - Monitor progress through real-time logging
   - Automatic early stopping and learning rate reduction
   - Save fine-tuned models for deployment

### Architecture Analysis and Optimization
- **Automated Analysis**: Use "Explain Architecture" to get detailed insights about your network design
- **Pattern Recognition**: Identify CNN-FC hybrids, RNN patterns, and optimization opportunities
- **Optimizer Comparison**: Access comprehensive guide comparing Adam, SGD, RMSprop performance characteristics
- **Gradient Monitoring**: Enable gradient visualization to detect vanishing/exploding gradients early

### GAN and Generative Models
1. Navigate to the **GAN & Advanced** tab
2. **Configure GAN Training**:
   - Select GAN type: Vanilla, DCGAN, Conditional GAN
   - Set latent dimension, batch size, and training epochs
   - Configure separate learning rates for generator and discriminator
3. **Framework Features**:
   - Current implementation provides infrastructure for GAN training
   - Generator and discriminator architecture templates
   - Adversarial training loop structure
   - Sample generation pipeline foundation

## 📊 Advanced Usage Examples

### Custom CNN Architecture for Image Classification
```python
# Example architecture design through GUI:
# Layer 1: Conv2D(filters=32, kernel_size=(3,3), activation='relu')
# Layer 2: BatchNormalization()
# Layer 3: MaxPooling2D(pool_size=(2,2))
# Layer 4: Conv2D(filters=64, kernel_size=(3,3), activation='relu')
# Layer 5: GlobalAveragePooling2D()
# Layer 6: Dense(units=128, activation='relu')
# Layer 7: Dropout(rate=0.5)
# Layer 8: Dense(units=num_classes, activation='softmax')
```

### Transfer Learning Best Practices
- **Start with frozen base layers** for initial training
- **Use low learning rates** (1e-4 to 1e-5) for fine-tuning
- **Gradually unfreeze layers** for progressive fine-tuning
- **Apply data augmentation** to prevent overfitting on small datasets
- **Monitor validation metrics** closely to detect overfitting

### Optimizer Selection Guidelines
- **Adam**: Best general-purpose optimizer, adaptive learning rates, good for most tasks
- **SGD with Momentum**: Better final convergence, requires careful learning rate tuning
- **RMSprop**: Excellent for RNNs, handles sparse gradients effectively
- **Learning Rate Scheduling**: Use step decay or exponential decay for improved convergence

## 🖥️ Development Environment

This enhanced GUI framework has been developed and tested in the following environment:

- **Operating System**: Windows 11 IoT Enterprise LTSC with WSL2 (Ubuntu 24.04.01 LTS)
- **Python**: 3.12.3 within a Virtual Environment (venv)
- **Hardware**:
  - **CPU**: Intel i7-11370H
  - **GPU**: NVIDIA RTX 3050 Ti with Studio Driver 572.83
- **ML Framework Configuration**:
  - CUDA 12.8
  - cuDNN 9.8.0
  - TensorRT 10.9
- **IDE**: Visual Studio Code

## ⚠️ Known Issues and Solutions

### Qt/XCB Plugin Loading Issues (WSL2 + GPU)

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

### Memory Management for Large Models
- **GPU Memory**: Large pre-trained models may require 4GB+ GPU memory
- **System RAM**: Recommend 8GB+ for smooth operation with multiple models
- **Model Loading**: First-time pre-trained model downloads may take several minutes
- **Training Monitoring**: Gradient monitoring increases memory usage slightly

### Dataset Compatibility
- **Image Data**: Automatically resized and preprocessed for pre-trained models
- **Tabular Data**: Best suited for classical ML algorithms and fully connected networks
- **Sequence Data**: Requires proper reshaping for LSTM/GRU layers
- **Custom Datasets**: CSV files with clear target column identification

## 🆕 What's New in v0.1.3

### Major Feature Additions
- **🧠 Advanced Deep Learning Tab**: Complete neural network design environment with 15+ layer types
- **🚀 Transfer Learning Integration**: 9 pre-trained models with comprehensive fine-tuning controls
- **🔄 Real-time Training Monitoring**: Live logs, gradient visualization, and progress tracking
- **🎨 GAN Framework**: Infrastructure for generative adversarial network implementation
- **📊 Architecture Analysis**: Automated pattern recognition and optimization recommendations
- **⚡ Multi-threaded Training**: Non-blocking operations with responsive UI

### Enhanced User Experience
- **Visual Architecture Designer**: Tree-view representation of network layers with parameter display
- **Context-aware Parameter Inputs**: Dynamic forms that adapt to selected layer types
- **Professional Training Interface**: Real-time logs with syntax highlighting and auto-scrolling
- **Comprehensive Save/Load System**: Complete architecture and model persistence with metadata

### Technical Improvements
- **Thread-safe Training**: Background training with signal-based communication
- **Memory Optimization**: Efficient handling of large models and datasets
- **Error Handling**: Comprehensive exception handling with user-friendly error messages
- **Code Quality**: Enhanced documentation, type hints, and modular design

## 📚 Additional Resources

For further assistance with installation issues or for more detailed information about the dependencies, please refer to these official resources:

- **TensorFlow Installation**: [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip)
- **CUDA Installation**: [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- **cuDNN Installation**: [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/)
- **TensorRT Installation** (optional): [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- **CUDA-Enabled GPUs**: [List of CUDA-Enabled NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)
- **Transfer Learning Guide**: [TensorFlow Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
- **Keras Applications**: [Pre-trained Models Documentation](https://keras.io/api/applications/)

If you encounter any problems with running the program or installing prerequisites, the resources above should be your first reference point for troubleshooting.

## 📞 Support

For issues related to this implementation, please open an issue in this repository. For questions about the original framework, please refer to the [original repository](https://github.com/bayraktare/MKT3434_2025).

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
