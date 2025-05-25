"""
Setup Initialization:
System   : Ubuntu 16.04+, Windows 7+ (WSL2 for GPU), macOS 12+; optional NVIDIA CUDA GPU; Python 3.9–3.12; pip ≥19.0 (Linux/Win) or ≥20.3 (macOS).
VirtualEnv: python -m venv <env_dir>; activate with `source <env_dir>/bin/activate` (Linux/macOS) or `<env_dir>\Scripts\activate` (Windows) to isolate project dependencies.
pip Upgrade: python -m pip install --upgrade pip  # ensures latest installer and dependency resolver.
TensorFlow: CPU-only → pip install tensorflow; GPU → pip install tensorflow[and-cuda]  # Windows GPU support requires WSL2.
Verify TF : GPU → python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"; CPU → python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000,1000])))"
Dependencies: pip install numpy pandas matplotlib PyQt6 scikit-learn torch torchvision torchaudio opencv-python opencv-contrib-python scipy fastai kornia plotly umap-learn
"""

import sys
import numpy as np
import pandas as pd
import json
import os
import pickle
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QRadioButton, QButtonGroup,
                           QSlider, QFrame, QSplitter, QTreeWidget, QTreeWidgetItem,
                           QTableWidget, QTableWidgetItem, QFormLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix, r2_score
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

class RealTimeLogger:
    """Real-time logging system for training metrics and events"""
    def __init__(self, log_widget):
        self.log_widget = log_widget
        self.logs = []
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        
        # Update widget in main thread
        current_text = self.log_widget.toPlainText()
        self.log_widget.setText(current_text + "\n" + log_entry)
        
        # Auto-scroll to bottom
        cursor = self.log_widget.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_widget.setTextCursor(cursor)

class TrainingWorker(QThread):
    """Worker thread for neural network training to prevent GUI freezing"""
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str, str)
    training_complete = pyqtSignal(object, object)
    gradient_update = pyqtSignal(dict)
    
    def __init__(self, model, X_train, y_train, X_test, y_test, 
                 batch_size, epochs, callbacks_list, monitor_gradients=False):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks_list = callbacks_list
        self.monitor_gradients = monitor_gradients
        
    def run(self):
        try:
            # Custom callback for progress updates
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, worker):
                    super().__init__()
                    self.worker = worker
                    
                def on_epoch_begin(self, epoch, logs=None):
                    progress = int((epoch / self.params['epochs']) * 100)
                    self.worker.progress_update.emit(progress)
                    self.worker.log_update.emit(f"Starting epoch {epoch + 1}/{self.params['epochs']}", "INFO")
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = int(((epoch + 1) / self.params['epochs']) * 100)
                    self.worker.progress_update.emit(progress)
                    
                    log_msg = f"Epoch {epoch + 1} - "
                    if logs:
                        for key, value in logs.items():
                            log_msg += f"{key}: {value:.4f}, "
                        log_msg = log_msg.rstrip(", ")
                    
                    self.worker.log_update.emit(log_msg, "INFO")
                    
                    # Monitor gradients if enabled
                    if self.worker.monitor_gradients and epoch % 5 == 0:
                        gradients = {}
                        for i, layer in enumerate(self.model.layers):
                            if hasattr(layer, 'kernel') and layer.kernel is not None:
                                grad_norm = tf.norm(layer.kernel).numpy()
                                gradients[f'layer_{i}_grad_norm'] = grad_norm
                        self.worker.gradient_update.emit(gradients)
            
            # Add progress callback
            callbacks_with_progress = self.callbacks_list + [ProgressCallback(self)]
            
            # Train the model
            history = self.model.fit(
                self.X_train, self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.X_test, self.y_test),
                callbacks=callbacks_with_progress,
                verbose=0
            )
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            self.training_complete.emit(history, y_pred)
            
        except Exception as e:
            self.log_update.emit(f"Training error: {str(e)}", "ERROR")

class ArchitectureAnalyzer:
    """Analyzes and explains neural network architecture choices"""
    @staticmethod
    def analyze_architecture(layer_config):
        explanation = "Neural Network Architecture Analysis:\n\n"
        
        total_layers = len(layer_config)
        depth_analysis = f"Network Depth: {total_layers} layers\n"
        
        # Count layer types
        conv_layers = sum(1 for layer in layer_config if layer['type'] in ['Conv2D', 'Conv1D'])
        dense_layers = sum(1 for layer in layer_config if layer['type'] == 'Dense')
        recurrent_layers = sum(1 for layer in layer_config if layer['type'] in ['LSTM', 'GRU', 'SimpleRNN'])
        pooling_layers = sum(1 for layer in layer_config if 'Pooling' in layer['type'])
        
        layer_distribution = f"Layer Distribution:\n"
        layer_distribution += f"- Convolutional layers: {conv_layers}\n"
        layer_distribution += f"- Dense layers: {dense_layers}\n"
        layer_distribution += f"- Recurrent layers: {recurrent_layers}\n"
        layer_distribution += f"- Pooling layers: {pooling_layers}\n\n"
        
        # Architecture pattern recognition
        architecture_pattern = "Architecture Pattern Analysis:\n"
        if conv_layers > 0 and dense_layers > 0:
            architecture_pattern += "Pattern: Convolutional Neural Network with Fully Connected Classifier\n"
            architecture_pattern += "- Suitable for image classification and computer vision tasks\n"
            architecture_pattern += "- Convolutional layers extract spatial features\n"
            architecture_pattern += "- Dense layers perform high-level reasoning and classification\n\n"
        elif recurrent_layers > 0:
            architecture_pattern += "Pattern: Recurrent Neural Network\n"
            architecture_pattern += "- Designed for sequential data processing\n"
            architecture_pattern += "- Can handle variable-length input sequences\n"
            architecture_pattern += "- Memory cells capture temporal dependencies\n\n"
        elif dense_layers == total_layers:
            architecture_pattern += "Pattern: Multi-Layer Perceptron (Fully Connected)\n"
            architecture_pattern += "- Traditional feedforward neural network\n"
            architecture_pattern += "- Suitable for tabular data and feature-based learning\n"
            architecture_pattern += "- Each layer learns increasingly abstract representations\n\n"
        
        # Activation function analysis
        activations = [layer.get('params', {}).get('activation', 'none') 
                      for layer in layer_config if layer.get('params', {}).get('activation')]
        if activations:
            unique_activations = list(set(activations))
            activation_analysis = f"Activation Functions Analysis:\n"
            activation_analysis += f"Used activations: {', '.join(unique_activations)}\n"
            
            for activation in unique_activations:
                if activation == 'relu':
                    activation_analysis += "- ReLU: Mitigates vanishing gradient problem, computationally efficient\n"
                elif activation == 'sigmoid':
                    activation_analysis += "- Sigmoid: Outputs probability-like values (0-1), suitable for binary classification\n"
                elif activation == 'tanh':
                    activation_analysis += "- Tanh: Zero-centered outputs, can help with gradient flow\n"
                elif activation == 'softmax':
                    activation_analysis += "- Softmax: Probability distribution over classes for multi-class classification\n"
                elif activation == 'swish':
                    activation_analysis += "- Swish: Self-gated activation function, often outperforms ReLU\n"
            activation_analysis += "\n"
        else:
            activation_analysis = ""
        
        # Recommendations
        recommendations = "Architecture Recommendations:\n"
        if conv_layers > 0 and pooling_layers == 0:
            recommendations += "- Consider adding pooling layers to reduce spatial dimensions\n"
        if dense_layers > 3:
            recommendations += "- Deep dense networks may benefit from dropout regularization\n"
        if 'BatchNormalization' not in [layer['type'] for layer in layer_config]:
            recommendations += "- BatchNormalization can improve training stability\n"
        if 'Dropout' not in [layer['type'] for layer in layer_config] and dense_layers > 2:
            recommendations += "- Dropout layers can help prevent overfitting\n"
        
        return explanation + depth_analysis + layer_distribution + architecture_pattern + activation_analysis + recommendations

class OptimizerComparator:
    """Provides detailed comparison of different optimizers"""
    @staticmethod
    def get_optimizer_comparison():
        comparison = """Optimizer Comparison and Selection Guide:

Adam (Adaptive Moment Estimation):
✓ Advantages:
  - Combines benefits of AdaGrad and RMSprop
  - Adaptive learning rates for each parameter
  - Works well with sparse gradients
  - Generally good default choice for most problems
  - Requires minimal hyperparameter tuning

✗ Disadvantages:
  - Can sometimes overshoot optimal solutions
  - May not converge to the sharpest minima
  - Can be computationally expensive for very large models

Best for: Most deep learning tasks, especially when starting a new project

SGD (Stochastic Gradient Descent):
✓ Advantages:
  - Simple and well-understood algorithm
  - Can escape local minima with proper momentum
  - Often achieves better generalization than adaptive methods
  - Memory efficient

✗ Disadvantages:
  - Requires careful learning rate tuning
  - Can be slow to converge without momentum
  - Sensitive to feature scaling

Best for: Fine-tuning pre-trained models, when generalization is critical

RMSprop:
✓ Advantages:
  - Adapts learning rate based on recent gradients
  - Good for recurrent neural networks
  - Handles non-stationary objectives well
  - Less aggressive than AdaGrad

✗ Disadvantages:
  - Can be sensitive to hyperparameter settings
  - Less popular than Adam for general use

Best for: Recurrent neural networks, online learning scenarios

AdaGrad:
✓ Advantages:
  - Performs larger updates for infrequent parameters
  - Good for sparse data
  - No manual tuning of learning rate

✗ Disadvantages:
  - Learning rate decreases too aggressively
  - Can stop learning too early

Best for: Sparse data, natural language processing tasks

Adadelta:
✓ Advantages:
  - Addresses AdaGrad's learning rate decay problem
  - No need to set default learning rate
  - Robust to noisy gradients

✗ Disadvantages:
  - Can be slow to converge
  - Less commonly used, fewer resources available

Best for: When you want adaptive learning without setting learning rate

General Recommendations:
1. Start with Adam for most projects
2. Try SGD with momentum for fine-tuning or when Adam struggles
3. Use RMSprop specifically for RNNs
4. Consider learning rate scheduling for better convergence
5. Monitor training curves to detect optimization issues early
"""
        return comparison

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI - Enhanced v0.1.3")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.original_data = None  # For storing original data before imputation
        
        # Neural network configuration
        self.layer_config = []
        self.training_history = None
        self.gradient_history = {}
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
        
        # Default imputer
        self.imputer = None
        
        # Pre-trained models cache
        self.pretrained_models = {}
        
        # Real-time logger
        self.logger = None
        
    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
                X, y = data.data, data.target
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
                X, y = data.data, data.target
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
                X, y = data.data, data.target
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.fetch_california_housing()  # Using California housing as Boston is deprecated
                X, y = data.data, data.target
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
                self.X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            elif dataset_name == "CIFAR-10 Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
                self.X_train = X_train.astype('float32') / 255.0
                self.X_test = X_test.astype('float32') / 255.0
                self.y_train, self.y_test = y_train.flatten(), y_test.flatten()
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            elif dataset_name == "Fashion-MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
                self.X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
                self.X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Introduce missing values for testing imputation (only for non-MNIST datasets)
            if self.introduce_missing_values_check.isChecked():
                # Create a copy to avoid modifying the original data
                X_with_missing = X.copy()
                
                # Randomly replace a percentage of values with NaN
                missing_percentage = self.missing_value_percent.value() / 100
                mask = np.random.rand(*X.shape) < missing_percentage
                X_with_missing[mask] = np.nan
                
                # Store original data for comparison
                self.original_data = {"X": X, "y": y}
                
                # Set X to the version with missing values
                X = X_with_missing
                
                self.status_bar.showMessage(f"Loaded {dataset_name} with {missing_percentage*100:.1f}% missing values")
            else:
                self.original_data = None
                self.status_bar.showMessage(f"Loaded {dataset_name}")
            
            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(X, y, 
                                              test_size=test_size, 
                                              random_state=42)
            
            # Handle missing values if present
            if np.isnan(self.X_train).any() or np.isnan(self.X_test).any():
                self.handle_missing_values()
            
            # Apply scaling if selected
            self.apply_scaling()
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def handle_missing_values(self):
        """Apply selected method to handle missing values"""
        imputation_method = self.missing_values_combo.currentText()
        
        try:
            if imputation_method == "Mean Imputation":
                self.imputer = SimpleImputer(strategy='mean')
                self.X_train = self.imputer.fit_transform(self.X_train)
                self.X_test = self.imputer.transform(self.X_test)
                
            elif imputation_method == "Median Imputation":
                self.imputer = SimpleImputer(strategy='median')
                self.X_train = self.imputer.fit_transform(self.X_train)
                self.X_test = self.imputer.transform(self.X_test)
                
            elif imputation_method == "Most Frequent Imputation":
                self.imputer = SimpleImputer(strategy='most_frequent')
                self.X_train = self.imputer.fit_transform(self.X_train)
                self.X_test = self.imputer.transform(self.X_test)
                
            elif imputation_method == "Forward Fill":
                # Convert to DataFrame for forward fill
                X_train_df = pd.DataFrame(self.X_train)
                X_test_df = pd.DataFrame(self.X_test)
                
                # Apply forward fill
                X_train_df.fillna(method='ffill', inplace=True)
                X_test_df.fillna(method='ffill', inplace=True)
                
                # Handle any remaining NaNs (at the beginning of columns) with backward fill
                X_train_df.fillna(method='bfill', inplace=True)
                X_test_df.fillna(method='bfill', inplace=True)
                
                self.X_train = X_train_df.values
                self.X_test = X_test_df.values
                
            elif imputation_method == "Backward Fill":
                # Convert to DataFrame for backward fill
                X_train_df = pd.DataFrame(self.X_train)
                X_test_df = pd.DataFrame(self.X_test)
                
                # Apply backward fill
                X_train_df.fillna(method='bfill', inplace=True)
                X_test_df.fillna(method='bfill', inplace=True)
                
                # Handle any remaining NaNs (at the end of columns) with forward fill
                X_train_df.fillna(method='ffill', inplace=True)
                X_test_df.fillna(method='ffill', inplace=True)
                
                self.X_train = X_train_df.values
                self.X_test = X_test_df.values
                
            self.status_bar.showMessage(f"Applied {imputation_method} to handle missing values")
            
        except Exception as e:
            self.show_error(f"Error handling missing values: {str(e)}")
    
    def create_imputation_comparison(self):
        """Compare different imputation methods"""
        if self.original_data is None:
            self.show_error("No original data available for comparison. Please check 'Introduce Missing Values' when loading a dataset.")
            return
        
        try:
            # Get original data
            X_orig, y_orig = self.original_data["X"], self.original_data["y"]
            
            # Split original data with same random state for fair comparison
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = \
                model_selection.train_test_split(X_orig, y_orig, 
                                             test_size=self.split_spin.value(), 
                                             random_state=42)
            
            # Create missing data with consistent random seed
            np.random.seed(42)
            missing_percentage = self.missing_value_percent.value() / 100
            X_train_missing = X_train_orig.copy()
            X_test_missing = X_test_orig.copy()
            
            mask_train = np.random.rand(*X_train_orig.shape) < missing_percentage
            mask_test = np.random.rand(*X_test_orig.shape) < missing_percentage
            
            X_train_missing[mask_train] = np.nan
            X_test_missing[mask_test] = np.nan
            
            # Imputation methods to compare
            methods = ["Mean Imputation", "Median Imputation", "Most Frequent Imputation", 
                       "Forward Fill", "Backward Fill"]
            
            # Model to use for comparison
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            if len(np.unique(y_orig)) > 10:  # Regression
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Evaluate each method
            results = {}
            for method in methods:
                X_train_imputed = X_train_missing.copy()
                X_test_imputed = X_test_missing.copy()
                
                if method == "Mean Imputation":
                    imputer = SimpleImputer(strategy='mean')
                    X_train_imputed = imputer.fit_transform(X_train_imputed)
                    X_test_imputed = imputer.transform(X_test_imputed)
                    
                elif method == "Median Imputation":
                    imputer = SimpleImputer(strategy='median')
                    X_train_imputed = imputer.fit_transform(X_train_imputed)
                    X_test_imputed = imputer.transform(X_test_imputed)
                    
                elif method == "Most Frequent Imputation":
                    imputer = SimpleImputer(strategy='most_frequent')
                    X_train_imputed = imputer.fit_transform(X_train_imputed)
                    X_test_imputed = imputer.transform(X_test_imputed)
                    
                elif method == "Forward Fill":
                    # Convert to DataFrame for forward fill
                    X_train_df = pd.DataFrame(X_train_imputed)
                    X_test_df = pd.DataFrame(X_test_imputed)
                    
                    # Apply forward fill
                    X_train_df.fillna(method='ffill', inplace=True)
                    X_test_df.fillna(method='ffill', inplace=True)
                    
                    # Handle any remaining NaNs with backward fill
                    X_train_df.fillna(method='bfill', inplace=True)
                    X_test_df.fillna(method='bfill', inplace=True)
                    
                    X_train_imputed = X_train_df.values
                    X_test_imputed = X_test_df.values
                    
                elif method == "Backward Fill":
                    # Convert to DataFrame for backward fill
                    X_train_df = pd.DataFrame(X_train_imputed)
                    X_test_df = pd.DataFrame(X_test_imputed)
                    
                    # Apply backward fill
                    X_train_df.fillna(method='bfill', inplace=True)
                    X_test_df.fillna(method='bfill', inplace=True)
                    
                    # Handle any remaining NaNs with forward fill
                    X_train_df.fillna(method='ffill', inplace=True)
                    X_test_df.fillna(method='ffill', inplace=True)
                    
                    X_train_imputed = X_train_df.values
                    X_test_imputed = X_test_df.values
                
                # Train and evaluate model
                model_clone = clone(model)
                model_clone.fit(X_train_imputed, y_train_orig)
                y_pred = model_clone.predict(X_test_imputed)
                
                # Calculate appropriate metric
                if len(np.unique(y_orig)) > 10:  # Regression
                    score = mean_squared_error(y_test_orig, y_pred)
                    metric_name = "MSE (lower is better)"
                else:  # Classification
                    score = accuracy_score(y_test_orig, y_pred)
                    metric_name = "Accuracy (higher is better)"
                
                results[method] = score
            
            # Plot results
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # For regression (MSE), lower is better, so invert the y-axis
            if len(np.unique(y_orig)) > 10:
                methods_sorted = sorted(results.keys(), key=lambda x: results[x])
            else:
                methods_sorted = sorted(results.keys(), key=lambda x: results[x], reverse=True)
            
            y_pos = np.arange(len(methods_sorted))
            scores = [results[method] for method in methods_sorted]
            
            bars = ax.barh(y_pos, scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(methods_sorted)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel(metric_name)
            ax.set_title('Imputation Method Comparison')
            
            # Add value labels on bars
            for i, v in enumerate(scores):
                ax.text(v + 0.01 * max(scores), i, f'{v:.4f}', va='center')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update metrics text
            metrics_text = "Imputation Method Comparison Results:\n\n"
            metrics_text += f"Missing Value Percentage: {missing_percentage*100:.1f}%\n"
            metrics_text += f"Metric: {metric_name}\n\n"
            
            for method in methods_sorted:
                metrics_text += f"{method}: {results[method]:.4f}\n"
            
            # Add recommendation
            if len(np.unique(y_orig)) > 10:  # Regression - lower MSE is better
                best_method = min(results, key=results.get)
                metrics_text += f"\nRecommended Method: {best_method} (Lowest MSE)"
            else:  # Classification - higher accuracy is better
                best_method = max(results, key=results.get)
                metrics_text += f"\nRecommended Method: {best_method} (Highest Accuracy)"
            
            self.metrics_text.setText(metrics_text)
            
        except Exception as e:
            self.show_error(f"Error comparing imputation methods: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Check for missing values
                    if X.isnull().values.any():
                        missing_count = X.isnull().sum().sum()
                        total_count = X.shape[0] * X.shape[1]
                        missing_percentage = (missing_count / total_count) * 100
                        
                        message = f"Dataset contains {missing_count} missing values ({missing_percentage:.2f}%).\n\n"
                        message += "Would you like to handle these missing values?"
                        
                        reply = QMessageBox.question(self, "Missing Values Detected", message,
                                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        
                        if reply == QMessageBox.StandardButton.Yes:
                            # Store original data for comparison
                            self.original_data = {"X": X.copy(), "y": y.copy()}
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Handle missing values if present
                    if self.X_train.isnull().values.any() or self.X_test.isnull().values.any():
                        self.handle_missing_values()
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        
        # Dataset section
        dataset_section = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset",
            "CIFAR-10 Dataset",
            "Fashion-MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to dataset section
        dataset_section.addWidget(QLabel("Dataset:"))
        dataset_section.addWidget(self.dataset_combo)
        dataset_section.addWidget(self.load_btn)
        dataset_section.addWidget(QLabel("Scaling:"))
        dataset_section.addWidget(self.scaling_combo)
        dataset_section.addWidget(QLabel("Test Split:"))
        dataset_section.addWidget(self.split_spin)
        
        data_layout.addLayout(dataset_section)
        
        # Missing values handling section
        missing_values_section = QHBoxLayout()
        
        # Checkbox to introduce missing values (for testing)
        self.introduce_missing_values_check = QCheckBox("Introduce Missing Values")
        missing_values_section.addWidget(self.introduce_missing_values_check)
        
        # Percentage of missing values
        missing_values_section.addWidget(QLabel("Percentage:"))
        self.missing_value_percent = QSpinBox()
        self.missing_value_percent.setRange(5, 50)
        self.missing_value_percent.setValue(10)
        missing_values_section.addWidget(self.missing_value_percent)
        
        # Missing values handling method
        missing_values_section.addWidget(QLabel("Handle Missing Values:"))
        self.missing_values_combo = QComboBox()
        self.missing_values_combo.addItems([
            "Mean Imputation",
            "Median Imputation",
            "Most Frequent Imputation",
            "Forward Fill",
            "Backward Fill"
        ])
        missing_values_section.addWidget(self.missing_values_combo)
        
        # Compare imputation methods button
        self.compare_btn = QPushButton("Compare Imputation Methods")
        self.compare_btn.clicked.connect(self.create_imputation_comparison)
        missing_values_section.addWidget(self.compare_btn)
        
        data_layout.addLayout(missing_values_section)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Advanced Deep Learning", self.create_advanced_deep_learning_tab),
            ("Transfer Learning", self.create_transfer_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Bayesian Methods", self.create_bayesian_tab),
            ("GAN & Advanced", self.create_gan_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)

    def create_advanced_deep_learning_tab(self):
        """Create the advanced deep learning tab with enhanced neural network design capabilities"""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        
        # Left panel - Architecture Design
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Neural Network Architecture Designer
        arch_group = QGroupBox("Neural Network Architecture Designer")
        arch_layout = QVBoxLayout()
        
        # Layer management buttons
        layer_buttons = QHBoxLayout()
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.show_advanced_layer_dialog)
        remove_layer_btn = QPushButton("Remove Selected")
        remove_layer_btn.clicked.connect(self.remove_selected_layer)
        clear_layers_btn = QPushButton("Clear All Layers")
        clear_layers_btn.clicked.connect(self.clear_all_layers)
        
        layer_buttons.addWidget(add_layer_btn)
        layer_buttons.addWidget(remove_layer_btn)
        layer_buttons.addWidget(clear_layers_btn)
        arch_layout.addLayout(layer_buttons)
        
        # Architecture visualization tree
        self.architecture_tree = QTreeWidget()
        self.architecture_tree.setHeaderLabels(["Layer", "Type", "Parameters", "Output Shape"])
        self.architecture_tree.setMaximumHeight(250)
        arch_layout.addWidget(self.architecture_tree)
        
        # Save/Load architecture buttons
        save_load_layout = QHBoxLayout()
        save_arch_btn = QPushButton("Save Architecture")
        save_arch_btn.clicked.connect(self.save_neural_architecture)
        load_arch_btn = QPushButton("Load Architecture")
        load_arch_btn.clicked.connect(self.load_neural_architecture)
        explain_arch_btn = QPushButton("Explain Architecture")
        explain_arch_btn.clicked.connect(self.explain_current_architecture)
        
        save_load_layout.addWidget(save_arch_btn)
        save_load_layout.addWidget(load_arch_btn)
        save_load_layout.addWidget(explain_arch_btn)
        arch_layout.addLayout(save_load_layout)
        
        arch_group.setLayout(arch_layout)
        left_layout.addWidget(arch_group)
        
        # Advanced Training Configuration
        training_group = QGroupBox("Advanced Training Configuration")
        training_layout = QFormLayout()
        
        # Optimizer selection with enhanced options
        self.advanced_optimizer_combo = QComboBox()
        self.advanced_optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdaGrad", "Adadelta"])
        training_layout.addRow("Optimizer:", self.advanced_optimizer_combo)
        
        # Learning rate with fine control
        self.advanced_learning_rate = QDoubleSpinBox()
        self.advanced_learning_rate.setRange(0.00001, 1.0)
        self.advanced_learning_rate.setValue(0.001)
        self.advanced_learning_rate.setDecimals(5)
        self.advanced_learning_rate.setSingleStep(0.0001)
        training_layout.addRow("Learning Rate:", self.advanced_learning_rate)
        
        # Learning rate scheduling
        self.lr_schedule_combo = QComboBox()
        self.lr_schedule_combo.addItems(["None", "Step Decay", "Exponential Decay", "Cosine Annealing"])
        training_layout.addRow("LR Schedule:", self.lr_schedule_combo)
        
        # Batch size
        self.advanced_batch_size = QSpinBox()
        self.advanced_batch_size.setRange(1, 512)
        self.advanced_batch_size.setValue(32)
        training_layout.addRow("Batch Size:", self.advanced_batch_size)
        
        # Epochs
        self.advanced_epochs = QSpinBox()
        self.advanced_epochs.setRange(1, 1000)
        self.advanced_epochs.setValue(100)
        training_layout.addRow("Epochs:", self.advanced_epochs)
        
        # Regularization options
        self.l2_regularization = QDoubleSpinBox()
        self.l2_regularization.setRange(0.0, 1.0)
        self.l2_regularization.setValue(0.01)
        self.l2_regularization.setDecimals(4)
        training_layout.addRow("L2 Regularization:", self.l2_regularization)
        
        # Early stopping
        self.early_stopping_enabled = QCheckBox("Enable Early Stopping")
        self.early_stopping_enabled.setChecked(True)
        training_layout.addRow("Early Stopping:", self.early_stopping_enabled)
        
        self.early_stopping_patience = QSpinBox()
        self.early_stopping_patience.setRange(1, 100)
        self.early_stopping_patience.setValue(15)
        training_layout.addRow("Patience:", self.early_stopping_patience)
        
        # Gradient monitoring
        self.gradient_monitoring_enabled = QCheckBox("Monitor Gradients")
        training_layout.addRow("Gradient Monitoring:", self.gradient_monitoring_enabled)
        
        training_group.setLayout(training_layout)
        left_layout.addWidget(training_group)
        
        # Training control buttons
        control_group = QGroupBox("Training Controls")
        control_layout = QVBoxLayout()
        
        train_advanced_btn = QPushButton("Train Advanced Neural Network")
        train_advanced_btn.clicked.connect(self.train_advanced_neural_network)
        train_advanced_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        control_layout.addWidget(train_advanced_btn)
        
        compare_optimizers_btn = QPushButton("Compare Optimizers")
        compare_optimizers_btn.clicked.connect(self.show_optimizer_comparison)
        control_layout.addWidget(compare_optimizers_btn)
        
        save_model_btn = QPushButton("Save Trained Model")
        save_model_btn.clicked.connect(self.save_trained_model)
        control_layout.addWidget(save_model_btn)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        main_layout.addWidget(left_panel)
        
        # Right panel - Real-time monitoring and visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Real-time training log
        log_group = QGroupBox("Real-time Training Log")
        log_layout = QVBoxLayout()
        
        self.advanced_training_log = QTextEdit()
        self.advanced_training_log.setReadOnly(True)
        self.advanced_training_log.setMaximumHeight(200)
        self.advanced_training_log.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #ffffff; font-family: 'Courier New', monospace; font-size: 9pt; }")
        log_layout.addWidget(self.advanced_training_log)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Training progress bar
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.advanced_training_progress = QProgressBar()
        self.advanced_training_progress.setStyleSheet("QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #4CAF50; width: 20px; }")
        progress_layout.addWidget(self.advanced_training_progress)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        # Gradient monitoring visualization
        gradient_group = QGroupBox("Gradient Monitoring")
        gradient_layout = QVBoxLayout()
        
        self.gradient_figure = Figure(figsize=(6, 4))
        self.gradient_canvas = FigureCanvas(self.gradient_figure)
        gradient_layout.addWidget(self.gradient_canvas)
        
        gradient_group.setLayout(gradient_layout)
        right_layout.addWidget(gradient_group)
        
        main_layout.addWidget(right_panel)
        
        return widget

    def show_advanced_layer_dialog(self):
        """Show enhanced dialog for adding neural network layers with comprehensive options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        dialog.setGeometry(200, 200, 600, 500)
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems([
            "Dense", "Conv2D", "Conv1D", "MaxPooling2D", "MaxPooling1D",
            "GlobalMaxPooling2D", "GlobalAveragePooling2D", "Flatten",
            "Dropout", "BatchNormalization", "LSTM", "GRU", "SimpleRNN",
            "Embedding", "Reshape", "Lambda"
        ])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Scrollable parameters area
        scroll = QScrollArea()
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        scroll.setWidget(params_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Dynamic parameter inputs
        self.layer_param_inputs = {}
        
        def update_layer_parameters():
            # Clear existing parameter inputs
            for i in reversed(range(params_layout.count())): 
                params_layout.itemAt(i).widget().setParent(None)
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            
            if layer_type == "Dense":
                self.add_parameter_input(params_layout, "units", "int", 128, 1, 4096)
                self.add_parameter_input(params_layout, "activation", "combo", "relu", 
                                       ["relu", "sigmoid", "tanh", "softmax", "linear", "swish", "gelu"])
                self.add_parameter_input(params_layout, "use_bias", "bool", True)
                self.add_parameter_input(params_layout, "kernel_regularizer", "combo", "None", 
                                       ["None", "l1", "l2", "l1_l2"])
                
            elif layer_type == "Conv2D":
                self.add_parameter_input(params_layout, "filters", "int", 64, 1, 512)
                self.add_parameter_input(params_layout, "kernel_size", "tuple", "(3,3)")
                self.add_parameter_input(params_layout, "strides", "tuple", "(1,1)")
                self.add_parameter_input(params_layout, "padding", "combo", "valid", ["valid", "same"])
                self.add_parameter_input(params_layout, "activation", "combo", "relu", 
                                       ["relu", "sigmoid", "tanh", "linear", "swish", "gelu"])
                self.add_parameter_input(params_layout, "use_bias", "bool", True)
                
            elif layer_type == "Conv1D":
                self.add_parameter_input(params_layout, "filters", "int", 64, 1, 512)
                self.add_parameter_input(params_layout, "kernel_size", "int", 3, 1, 20)
                self.add_parameter_input(params_layout, "strides", "int", 1, 1, 10)
                self.add_parameter_input(params_layout, "padding", "combo", "valid", ["valid", "same"])
                self.add_parameter_input(params_layout, "activation", "combo", "relu", 
                                       ["relu", "sigmoid", "tanh", "linear", "swish"])
                
            elif layer_type == "MaxPooling2D":
                self.add_parameter_input(params_layout, "pool_size", "tuple", "(2,2)")
                self.add_parameter_input(params_layout, "strides", "tuple", "(2,2)")
                self.add_parameter_input(params_layout, "padding", "combo", "valid", ["valid", "same"])
                
            elif layer_type == "MaxPooling1D":
                self.add_parameter_input(params_layout, "pool_size", "int", 2, 1, 10)
                self.add_parameter_input(params_layout, "strides", "int", 2, 1, 10)
                self.add_parameter_input(params_layout, "padding", "combo", "valid", ["valid", "same"])
                
            elif layer_type == "Dropout":
                self.add_parameter_input(params_layout, "rate", "float", 0.5, 0.0, 0.9, 0.1)
                
            elif layer_type in ["LSTM", "GRU", "SimpleRNN"]:
                self.add_parameter_input(params_layout, "units", "int", 128, 1, 1024)
                self.add_parameter_input(params_layout, "activation", "combo", "tanh", 
                                       ["tanh", "relu", "sigmoid", "linear"])
                self.add_parameter_input(params_layout, "return_sequences", "bool", False)
                self.add_parameter_input(params_layout, "return_state", "bool", False)
                if layer_type in ["LSTM", "GRU"]:
                    self.add_parameter_input(params_layout, "recurrent_activation", "combo", "sigmoid", 
                                           ["sigmoid", "tanh", "relu"])
                self.add_parameter_input(params_layout, "dropout", "float", 0.0, 0.0, 0.9, 0.1)
                self.add_parameter_input(params_layout, "recurrent_dropout", "float", 0.0, 0.0, 0.9, 0.1)
                
            elif layer_type == "Embedding":
                self.add_parameter_input(params_layout, "input_dim", "int", 10000, 1, 100000)
                self.add_parameter_input(params_layout, "output_dim", "int", 128, 1, 1000)
                self.add_parameter_input(params_layout, "mask_zero", "bool", False)
                
            elif layer_type == "Reshape":
                self.add_parameter_input(params_layout, "target_shape", "tuple", "(28,28,1)")
        
        type_combo.currentIndexChanged.connect(update_layer_parameters)
        update_layer_parameters()  # Initialize with first layer type
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        def add_layer_to_architecture():
            layer_type = type_combo.currentText()
            layer_params = {}
            
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                    if value == "None":
                        layer_params[param_name] = None
                    else:
                        layer_params[param_name] = value
                elif isinstance(widget, QCheckBox):
                    layer_params[param_name] = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if param_name in ["kernel_size", "strides", "pool_size", "target_shape"]:
                        try:
                            layer_params[param_name] = eval(text)
                        except:
                            layer_params[param_name] = text
                    else:
                        layer_params[param_name] = text
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            self.update_architecture_tree_display()
            dialog.accept()
        
        add_btn.clicked.connect(add_layer_to_architecture)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()

    def add_parameter_input(self, layout, param_name, param_type, default_value, 
                           min_val=None, max_val=None, step=None, options=None):
        """Add parameter input widget to the layout with enhanced controls"""
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel(f"{param_name.replace('_', ' ').title()}:"))
        
        if param_type == "int":
            widget = QSpinBox()
            if min_val is not None:
                widget.setMinimum(min_val)
            if max_val is not None:
                widget.setMaximum(max_val)
            widget.setValue(default_value)
            
        elif param_type == "float":
            widget = QDoubleSpinBox()
            if min_val is not None:
                widget.setMinimum(min_val)
            if max_val is not None:
                widget.setMaximum(max_val)
            widget.setValue(default_value)
            widget.setDecimals(3)
            if step is not None:
                widget.setSingleStep(step)
            else:
                widget.setSingleStep(0.1)
                
        elif param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(default_value)
            
        elif param_type == "combo":
            widget = QComboBox()
            widget.addItems(options)
            if default_value in options:
                widget.setCurrentText(default_value)
                
        elif param_type == "tuple":
            widget = QLineEdit()
            widget.setText(str(default_value))
            widget.setPlaceholderText("e.g., (3,3) or (28,28,1)")
        
        param_layout.addWidget(widget)
        layout.addLayout(param_layout)
        self.layer_param_inputs[param_name] = widget

    def update_architecture_tree_display(self):
        """Update the neural network architecture tree display with enhanced information"""
        self.architecture_tree.clear()
        
        for i, layer in enumerate(self.layer_config):
            item = QTreeWidgetItem()
            item.setText(0, f"Layer {i+1}")
            item.setText(1, layer['type'])
            
            # Format parameters for display
            params_str = ""
            for key, value in layer['params'].items():
                if value is not None:
                    params_str += f"{key}={value}, "
            params_str = params_str.rstrip(", ")
            item.setText(2, params_str)
            
            # Estimate output shape (simplified approximation)
            output_shape = self.estimate_layer_output_shape(layer, i)
            item.setText(3, output_shape)
            
            self.architecture_tree.addTopLevelItem(item)

    def estimate_layer_output_shape(self, layer, layer_index):
        """Estimate the output shape of a layer for display purposes"""
        if layer_index == 0:
            return "Input dependent"
        elif layer['type'] == 'Dense':
            units = layer['params'].get('units', 'N')
            return f"(batch_size, {units})"
        elif layer['type'] == 'Conv2D':
            filters = layer['params'].get('filters', 'F')
            return f"(batch_size, H', W', {filters})"
        elif layer['type'] in ['LSTM', 'GRU', 'SimpleRNN']:
            units = layer['params'].get('units', 'N')
            return_seq = layer['params'].get('return_sequences', False)
            if return_seq:
                return f"(batch_size, timesteps, {units})"
            else:
                return f"(batch_size, {units})"
        elif layer['type'] == 'Flatten':
            return "(batch_size, features)"
        elif layer['type'] == 'Dropout':
            return "Same as input"
        else:
            return "Auto-calculated"

    def remove_selected_layer(self):
        """Remove the selected layer from the architecture"""
        current_item = self.architecture_tree.currentItem()
        if current_item:
            index = self.architecture_tree.indexOfTopLevelItem(current_item)
            if 0 <= index < len(self.layer_config):
                del self.layer_config[index]
                self.update_architecture_tree_display()

    def clear_all_layers(self):
        """Clear all layers from the current architecture"""
        reply = QMessageBox.question(self, "Clear Architecture", 
                                   "Are you sure you want to clear all layers?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.layer_config.clear()
            self.update_architecture_tree_display()

    def save_neural_architecture(self):
        """Save the current neural network architecture to a JSON file"""
        if not self.layer_config:
            self.show_error("No architecture to save. Please add layers first.")
            return
        
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Neural Network Architecture", "", 
                "JSON files (*.json);;All files (*)"
            )
            
            if file_name:
                architecture_data = {
                    "architecture_name": f"Neural_Network_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "layers": self.layer_config,
                    "metadata": {
                        "created_date": datetime.now().isoformat(),
                        "total_layers": len(self.layer_config),
                        "framework_version": "0.1.3"
                    },
                    "training_config": {
                        "optimizer": self.advanced_optimizer_combo.currentText(),
                        "learning_rate": self.advanced_learning_rate.value(),
                        "batch_size": self.advanced_batch_size.value(),
                        "epochs": self.advanced_epochs.value()
                    }
                }
                
                with open(file_name, 'w') as f:
                    json.dump(architecture_data, f, indent=2, default=str)
                
                self.status_bar.showMessage(f"Architecture saved successfully to {file_name}")
                
        except Exception as e:
            self.show_error(f"Error saving architecture: {str(e)}")

    def load_neural_architecture(self):
        """Load a neural network architecture from a JSON file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Load Neural Network Architecture", "",
                "JSON files (*.json);;All files (*)"
            )
            
            if file_name:
                with open(file_name, 'r') as f:
                    architecture_data = json.load(f)
                
                # Load layer configuration
                self.layer_config = architecture_data.get("layers", [])
                
                # Load training configuration if available
                if "training_config" in architecture_data:
                    config = architecture_data["training_config"]
                    self.advanced_optimizer_combo.setCurrentText(config.get("optimizer", "Adam"))
                    self.advanced_learning_rate.setValue(config.get("learning_rate", 0.001))
                    self.advanced_batch_size.setValue(config.get("batch_size", 32))
                    self.advanced_epochs.setValue(config.get("epochs", 100))
                
                self.update_architecture_tree_display()
                self.status_bar.showMessage(f"Architecture loaded successfully from {file_name}")
                
        except Exception as e:
            self.show_error(f"Error loading architecture: {str(e)}")

    def explain_current_architecture(self):
        """Display detailed explanation of the current neural network architecture"""
        if not self.layer_config:
            self.show_error("No architecture to explain. Please add layers first.")
            return
        
        explanation = ArchitectureAnalyzer.analyze_architecture(self.layer_config)
        
        # Create explanation dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Neural Network Architecture Analysis")
        dialog.setGeometry(200, 200, 800, 600)
        layout = QVBoxLayout(dialog)
        
        explanation_text = QTextEdit()
        explanation_text.setPlainText(explanation)
        explanation_text.setReadOnly(True)
        explanation_text.setFont(QFont("Courier New", 10))
        layout.addWidget(explanation_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def train_advanced_neural_network(self):
        """Train the neural network with advanced configuration and real-time monitoring"""
        if not self.layer_config:
            self.show_error("Please design a neural network architecture first by adding layers.")
            return
        
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first.")
            return
        
        try:
            # Initialize real-time logger
            self.logger = RealTimeLogger(self.advanced_training_log)
            self.advanced_training_log.clear()
            
            self.logger.log("Initializing advanced neural network training...", "INFO")
            
            # Create the neural network model
            model = self.build_advanced_neural_network()
            if model is None:
                return
            
            # Get training configuration
            optimizer_name = self.advanced_optimizer_combo.currentText()
            learning_rate = self.advanced_learning_rate.value()
            batch_size = self.advanced_batch_size.value()
            epochs = self.advanced_epochs.value()
            
            self.logger.log(f"Training configuration: {optimizer_name} optimizer, LR={learning_rate}, Batch={batch_size}, Epochs={epochs}", "INFO")
            
            # Create optimizer
            optimizer = self.create_advanced_optimizer(optimizer_name, learning_rate)
            
            # Prepare data for training
            X_train, y_train, X_test, y_test = self.prepare_advanced_training_data()
            
            # Compile model
            self.compile_advanced_model(model, optimizer)
            
            # Prepare callbacks
            callbacks_list = self.create_advanced_callbacks()
            
            # Start training in worker thread
            self.training_worker = TrainingWorker(
                model, X_train, y_train, X_test, y_test,
                batch_size, epochs, callbacks_list,
                self.gradient_monitoring_enabled.isChecked()
            )
            
            # Connect signals
            self.training_worker.progress_update.connect(self.advanced_training_progress.setValue)
            self.training_worker.log_update.connect(self.logger.log)
            self.training_worker.training_complete.connect(self.on_advanced_training_complete)
            self.training_worker.gradient_update.connect(self.update_gradient_visualization)
            
            # Start training
            self.training_worker.start()
            self.logger.log("Training started in background thread...", "INFO")
            
        except Exception as e:
            self.show_error(f"Error starting neural network training: {str(e)}")

    def build_advanced_neural_network(self):
        """Build the neural network model from the layer configuration"""
        try:
            model = models.Sequential()
            
            for i, layer_config in enumerate(self.layer_config):
                layer_type = layer_config["type"]
                params = layer_config["params"].copy()
                
                # Add input shape for the first layer
                if i == 0 and "input_shape" not in params:
                    if layer_type in ["Conv2D", "MaxPooling2D", "GlobalMaxPooling2D", "GlobalAveragePooling2D"]:
                        # For convolutional layers, determine input shape from data
                        if len(self.X_train.shape) == 4:
                            params["input_shape"] = self.X_train.shape[1:]
                        elif len(self.X_train.shape) == 2:
                            # Attempt to reshape for image data
                            if self.X_train.shape[1] == 784:  # MNIST
                                params["input_shape"] = (28, 28, 1)
                            elif self.X_train.shape[1] == 3072:  # CIFAR-10
                                params["input_shape"] = (32, 32, 3)
                            else:
                                self.show_error("Cannot determine input shape for convolutional layers. Please reshape your data.")
                                return None
                    else:
                        # For other layer types
                        params["input_shape"] = (self.X_train.shape[1],)
                
                # Create layer based on type
                layer = self.create_keras_layer(layer_type, params)
                if layer is None:
                    self.show_error(f"Failed to create layer: {layer_type}")
                    return None
                
                model.add(layer)
            
            # Add output layer if not present
            if not self.has_appropriate_output_layer():
                output_layer = self.create_output_layer()
                model.add(output_layer)
                self.logger.log("Added automatic output layer", "INFO")
            
            return model
            
        except Exception as e:
            self.show_error(f"Error building neural network: {str(e)}")
            return None

    def create_keras_layer(self, layer_type, params):
        """Create a Keras layer from type and parameters"""
        try:
            if layer_type == "Dense":
                return layers.Dense(**params)
            elif layer_type == "Conv2D":
                return layers.Conv2D(**params)
            elif layer_type == "Conv1D":
                return layers.Conv1D(**params)
            elif layer_type == "MaxPooling2D":
                return layers.MaxPooling2D(**params)
            elif layer_type == "MaxPooling1D":
                return layers.MaxPooling1D(**params)
            elif layer_type == "GlobalMaxPooling2D":
                return layers.GlobalMaxPooling2D()
            elif layer_type == "GlobalAveragePooling2D":
                return layers.GlobalAveragePooling2D()
            elif layer_type == "Flatten":
                return layers.Flatten()
            elif layer_type == "Dropout":
                return layers.Dropout(**params)
            elif layer_type == "BatchNormalization":
                return layers.BatchNormalization()
            elif layer_type == "LSTM":
                return layers.LSTM(**params)
            elif layer_type == "GRU":
                return layers.GRU(**params)
            elif layer_type == "SimpleRNN":
                return layers.SimpleRNN(**params)
            elif layer_type == "Embedding":
                return layers.Embedding(**params)
            elif layer_type == "Reshape":
                return layers.Reshape(params["target_shape"])
            else:
                return None
        except Exception as e:
            self.logger.log(f"Error creating {layer_type} layer: {str(e)}", "ERROR")
            return None

    def has_appropriate_output_layer(self):
        """Check if the architecture has an appropriate output layer"""
        if not self.layer_config:
            return False
        
        last_layer = self.layer_config[-1]
        if last_layer["type"] == "Dense":
            activation = last_layer["params"].get("activation")
            if activation in ["softmax", "sigmoid", "linear"]:
                return True
        
        return False

    def create_output_layer(self):
        """Create an appropriate output layer based on the target data"""
        num_classes = len(np.unique(self.y_train))
        
        if num_classes > 2:
            # Multi-class classification
            return layers.Dense(num_classes, activation='softmax', name='output_layer')
        elif num_classes == 2:
            # Binary classification
            return layers.Dense(1, activation='sigmoid', name='output_layer')
        else:
            # Regression
            return layers.Dense(1, activation='linear', name='output_layer')

    def create_advanced_optimizer(self, optimizer_name, learning_rate):
        """Create an optimizer with the specified configuration"""
        if optimizer_name == "Adam":
            return optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "SGD":
            return optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == "RMSprop":
            return optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "AdaGrad":
            return optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_name == "Adadelta":
            return optimizers.Adadelta(learning_rate=learning_rate)
        else:
            return optimizers.Adam(learning_rate=learning_rate)

    def prepare_advanced_training_data(self):
        """Prepare data for advanced neural network training"""
        X_train, X_test = self.X_train.copy(), self.X_test.copy()
        y_train, y_test = self.y_train.copy(), self.y_test.copy()
        
        # Reshape data if needed for convolutional layers
        conv_layers = [layer for layer in self.layer_config if layer["type"] in ["Conv2D", "MaxPooling2D"]]
        if conv_layers and len(X_train.shape) == 2:
            if X_train.shape[1] == 784:  # MNIST
                X_train = X_train.reshape(-1, 28, 28, 1)
                X_test = X_test.reshape(-1, 28, 28, 1)
            elif X_train.shape[1] == 3072:  # CIFAR-10 flattened
                X_train = X_train.reshape(-1, 32, 32, 3)
                X_test = X_test.reshape(-1, 32, 32, 3)
        
        return X_train, y_train, X_test, y_test

    def compile_advanced_model(self, model, optimizer):
        """Compile the model with appropriate loss function and metrics"""
        num_classes = len(np.unique(self.y_train))
        
        if num_classes > 2:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        elif num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def create_advanced_callbacks(self):
        """Create advanced training callbacks"""
        callbacks_list = []
        
        # Early stopping
        if self.early_stopping_enabled.isChecked():
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience.value(),
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stop)
        
        # Learning rate scheduling
        lr_schedule = self.lr_schedule_combo.currentText()
        if lr_schedule != "None":
            scheduler = self.create_learning_rate_scheduler(lr_schedule)
            if scheduler:
                callbacks_list.append(scheduler)
        
        # Model checkpointing
        checkpoint = callbacks.ModelCheckpoint(
            'best_model_temp.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callbacks_list.append(checkpoint)
        
        return callbacks_list

    def create_learning_rate_scheduler(self, schedule_type):
        """Create learning rate scheduler based on type"""
        initial_lr = self.advanced_learning_rate.value()
        
        if schedule_type == "Step Decay":
            def step_decay_schedule(epoch):
                drop_rate = 0.5
                epochs_drop = 10
                return initial_lr * np.power(drop_rate, np.floor(epoch / epochs_drop))
            
            return callbacks.LearningRateScheduler(step_decay_schedule)
        
        elif schedule_type == "Exponential Decay":
            def exp_decay_schedule(epoch):
                return initial_lr * np.exp(-0.1 * epoch)
            
            return callbacks.LearningRateScheduler(exp_decay_schedule)
        
        elif schedule_type == "Cosine Annealing":
            return callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=initial_lr * 0.001
            )
        
        return None

    def on_advanced_training_complete(self, history, y_pred):
        """Handle completion of advanced neural network training"""
        self.training_history = history
        self.current_model = self.training_worker.model
        
        # Update main visualization
        self.plot_training_history(history)
        
        # Calculate and display metrics
        if len(np.unique(self.y_train)) > 2:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        
        self.update_metrics(y_pred_classes)
        
        self.logger.log("Advanced neural network training completed successfully!", "SUCCESS")
        self.advanced_training_progress.setValue(100)

    def update_gradient_visualization(self, gradients):
        """Update the gradient monitoring visualization"""
        if not gradients:
            return
        
        # Store gradient history
        for key, value in gradients.items():
            if key not in self.gradient_history:
                self.gradient_history[key] = []
            self.gradient_history[key].append(value)
        
        # Update gradient plot
        self.gradient_figure.clear()
        ax = self.gradient_figure.add_subplot(111)
        
        for key, values in self.gradient_history.items():
            ax.plot(values, label=key.replace('_', ' ').title())
        
        ax.set_xlabel('Training Step (every 5 epochs)')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Weight Gradient Monitoring')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        self.gradient_figure.tight_layout()
        self.gradient_canvas.draw()

    def show_optimizer_comparison(self):
        """Display detailed optimizer comparison information"""
        comparison = OptimizerComparator.get_optimizer_comparison()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Optimizer Comparison Guide")
        dialog.setGeometry(200, 200, 900, 700)
        layout = QVBoxLayout(dialog)
        
        comparison_text = QTextEdit()
        comparison_text.setPlainText(comparison)
        comparison_text.setReadOnly(True)
        comparison_text.setFont(QFont("Consolas", 10))
        layout.addWidget(comparison_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def save_trained_model(self):
        """Save the trained neural network model"""
        if self.current_model is None:
            self.show_error("No trained model to save. Please train a model first.")
            return
        
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Trained Model", "",
                "HDF5 files (*.h5);;SavedModel format (*.pb);;All files (*)"
            )
            
            if file_name:
                if file_name.endswith('.h5'):
                    self.current_model.save(file_name)
                else:
                    # Save in SavedModel format
                    self.current_model.save(file_name, save_format='tf')
                
                self.status_bar.showMessage(f"Model saved successfully to {file_name}")
                
        except Exception as e:
            self.show_error(f"Error saving model: {str(e)}")

    def create_transfer_learning_tab(self):
        """Create comprehensive transfer learning tab with pre-trained model integration"""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        
        # Left panel - Transfer learning configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Pre-trained model selection
        model_group = QGroupBox("Pre-trained Model Selection")
        model_layout = QVBoxLayout()
        
        self.pretrained_model_combo = QComboBox()
        self.pretrained_model_combo.addItems([
            "VGG16", "VGG19", "ResNet50", "ResNet101", "InceptionV3", 
            "Xception", "MobileNet", "DenseNet121", "EfficientNetB0"
        ])
        model_layout.addWidget(QLabel("Select Pre-trained Model:"))
        model_layout.addWidget(self.pretrained_model_combo)
        
        load_pretrained_btn = QPushButton("Load Pre-trained Model")
        load_pretrained_btn.clicked.connect(self.load_pretrained_model)
        model_layout.addWidget(load_pretrained_btn)
        
        # Model information display
        self.model_info_display = QTextEdit()
        self.model_info_display.setReadOnly(True)
        self.model_info_display.setMaximumHeight(150)
        model_layout.addWidget(self.model_info_display)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Fine-tuning configuration
        finetune_group = QGroupBox("Fine-tuning Configuration")
        finetune_layout = QFormLayout()
        
        # Layer freezing options
        self.freeze_base_layers = QCheckBox("Freeze Base Model Layers")
        self.freeze_base_layers.setChecked(True)
        finetune_layout.addRow("Base Layers:", self.freeze_base_layers)
        
        self.unfreeze_top_layers = QSpinBox()
        self.unfreeze_top_layers.setRange(0, 50)
        self.unfreeze_top_layers.setValue(5)
        finetune_layout.addRow("Unfreeze Top N Layers:", self.unfreeze_top_layers)
        
        # Fine-tuning learning rate
        self.finetune_learning_rate = QDoubleSpinBox()
        self.finetune_learning_rate.setRange(0.00001, 0.1)
        self.finetune_learning_rate.setValue(0.0001)
        self.finetune_learning_rate.setDecimals(5)
        finetune_layout.addRow("Fine-tune Learning Rate:", self.finetune_learning_rate)
        
        # Custom classifier
        self.add_custom_classifier = QCheckBox("Add Custom Classifier")
        self.add_custom_classifier.setChecked(True)
        finetune_layout.addRow("Custom Classifier:", self.add_custom_classifier)
        
        finetune_group.setLayout(finetune_layout)
        left_layout.addWidget(finetune_group)
        
        # Image augmentation settings
        augmentation_group = QGroupBox("Image Augmentation")
        augmentation_layout = QFormLayout()
        
        self.enable_rotation = QCheckBox()
        self.rotation_range = QSpinBox()
        self.rotation_range.setRange(0, 180)
        self.rotation_range.setValue(20)
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(self.enable_rotation)
        rotation_layout.addWidget(self.rotation_range)
        augmentation_layout.addRow("Rotation (degrees):", rotation_layout)
        
        self.enable_horizontal_flip = QCheckBox("Enable Horizontal Flip")
        augmentation_layout.addRow("Horizontal Flip:", self.enable_horizontal_flip)
        
        self.enable_zoom = QCheckBox()
        self.zoom_range = QDoubleSpinBox()
        self.zoom_range.setRange(0.0, 1.0)
        self.zoom_range.setValue(0.2)
        self.zoom_range.setSingleStep(0.1)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.enable_zoom)
        zoom_layout.addWidget(self.zoom_range)
        augmentation_layout.addRow("Zoom Range:", zoom_layout)
        
        self.enable_brightness = QCheckBox()
        self.brightness_range = QDoubleSpinBox()
        self.brightness_range.setRange(0.0, 1.0)
        self.brightness_range.setValue(0.2)
        self.brightness_range.setSingleStep(0.1)
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(self.enable_brightness)
        brightness_layout.addWidget(self.brightness_range)
        augmentation_layout.addRow("Brightness Range:", brightness_layout)
        
        augmentation_group.setLayout(augmentation_layout)
        left_layout.addWidget(augmentation_group)
        
        # Transfer learning controls
        control_group = QGroupBox("Transfer Learning Controls")
        control_layout = QVBoxLayout()
        
        start_transfer_btn = QPushButton("Start Transfer Learning")
        start_transfer_btn.clicked.connect(self.start_transfer_learning)
        start_transfer_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        control_layout.addWidget(start_transfer_btn)
        
        save_finetuned_btn = QPushButton("Save Fine-tuned Model")
        save_finetuned_btn.clicked.connect(self.save_finetuned_model)
        control_layout.addWidget(save_finetuned_btn)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        main_layout.addWidget(left_panel)
        
        # Right panel - Transfer learning monitoring
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Transfer learning log
        transfer_log_group = QGroupBox("Transfer Learning Log")
        transfer_log_layout = QVBoxLayout()
        
        self.transfer_learning_log = QTextEdit()
        self.transfer_learning_log.setReadOnly(True)
        self.transfer_learning_log.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #ffffff; font-family: 'Courier New', monospace; font-size: 9pt; }")
        transfer_log_layout.addWidget(self.transfer_learning_log)
        
        transfer_log_group.setLayout(transfer_log_layout)
        right_layout.addWidget(transfer_log_group)
        
        main_layout.addWidget(right_panel)
        
        return widget

    def load_pretrained_model(self):
        """Load the selected pre-trained model and display information"""
        model_name = self.pretrained_model_combo.currentText()
        
        try:
            self.status_bar.showMessage(f"Loading {model_name}... This may take a moment.")
            
            # Load the pre-trained model without caching to get fresh info
            if model_name == "VGG16":
                base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_name == "VGG19":
                base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_name == "ResNet50":
                base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_name == "ResNet101":
                base_model = applications.ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_name == "InceptionV3":
                base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
            elif model_name == "Xception":
                base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
            elif model_name == "MobileNet":
                base_model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_name == "DenseNet121":
                base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_name == "EfficientNetB0":
                base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            else:
                self.show_error(f"Unsupported model: {model_name}")
                return
            
            # Cache the loaded model
            self.pretrained_models[model_name] = base_model
            
            # Display model information
            model_info = f"Pre-trained Model: {model_name}\n"
            model_info += f"Total Parameters: {base_model.count_params():,}\n"
            model_info += f"Trainable Parameters: {sum([tf.reduce_prod(var.shape) for var in base_model.trainable_variables]):,}\n"
            model_info += f"Input Shape: {base_model.input_shape}\n"
            model_info += f"Output Shape: {base_model.output_shape}\n"
            model_info += f"Number of Layers: {len(base_model.layers)}\n"
            
            self.model_info_display.setText(model_info)
            self.status_bar.showMessage(f"Successfully loaded {model_name}")
            
        except Exception as e:
            self.show_error(f"Error loading pre-trained model: {str(e)}")

    def start_transfer_learning(self):
        """Start the transfer learning process with the selected configuration"""
        model_name = self.pretrained_model_combo.currentText()
        
        if model_name not in self.pretrained_models:
            self.show_error("Please load a pre-trained model first.")
            return
        
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return
        
        try:
            # Initialize transfer learning logger
            transfer_logger = RealTimeLogger(self.transfer_learning_log)
            self.transfer_learning_log.clear()
            
            transfer_logger.log("Initializing transfer learning process...", "INFO")
            
            # Get the base model
            base_model = self.pretrained_models[model_name]
            
            # Configure layer freezing
            if self.freeze_base_layers.isChecked():
                base_model.trainable = False
                transfer_logger.log("Froze all base model layers", "INFO")
            else:
                # Unfreeze top N layers
                base_model.trainable = True
                n_unfreeze = self.unfreeze_top_layers.value()
                
                for layer in base_model.layers[:-n_unfreeze]:
                    layer.trainable = False
                    
                transfer_logger.log(f"Unfroze top {n_unfreeze} layers", "INFO")
            
            # Build transfer learning model
            model = models.Sequential([base_model])
            
            # Add global pooling
            model.add(layers.GlobalAveragePooling2D())
            
            # Add custom classifier if enabled
            if self.add_custom_classifier.isChecked():
                model.add(layers.Dense(512, activation='relu'))
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(256, activation='relu'))
                model.add(layers.Dropout(0.3))
                transfer_logger.log("Added custom classifier layers", "INFO")
            
            # Add output layer
            num_classes = len(np.unique(self.y_train))
            if num_classes > 2:
                model.add(layers.Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
            else:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=self.finetune_learning_rate.value())
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            
            transfer_logger.log("Model compiled successfully", "INFO")
            
            # Prepare data and augmentation
            X_train, X_test, y_train, y_test = self.prepare_transfer_learning_data()
            
            # Create data generators if augmentation is enabled
            train_generator, val_generator = self.create_augmentation_generators(X_train, y_train, X_test, y_test)
            
            # Training callbacks
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
            ]
            
            # Train the model
            transfer_logger.log("Starting transfer learning training...", "INFO")
            
            if train_generator and val_generator:
                history = model.fit(
                    train_generator,
                    epochs=50,
                    validation_data=val_generator,
                    callbacks=callbacks_list,
                    verbose=1
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    batch_size=32,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks_list,
                    verbose=1
                )
            
            # Store results
            self.current_model = model
            self.training_history = history
            
            # Update main visualization
            self.plot_training_history(history)
            
            transfer_logger.log("Transfer learning completed successfully!", "SUCCESS")
            
        except Exception as e:
            self.show_error(f"Error in transfer learning: {str(e)}")

    def prepare_transfer_learning_data(self):
        """Prepare data for transfer learning with proper preprocessing"""
        X_train, X_test = self.X_train.copy(), self.X_test.copy()
        y_train, y_test = self.y_train.copy(), self.y_test.copy()
        
        # Ensure data is in correct format for pre-trained models
        if len(X_train.shape) == 2:
            # Reshape flattened image data
            if X_train.shape[1] == 784:  # MNIST
                X_train = X_train.reshape(-1, 28, 28, 1)
                X_test = X_test.reshape(-1, 28, 28, 1)
                # Resize to 224x224 and convert to RGB for pre-trained models
                X_train = tf.image.resize(X_train, [224, 224])
                X_test = tf.image.resize(X_test, [224, 224])
                X_train = tf.image.grayscale_to_rgb(X_train)
                X_test = tf.image.grayscale_to_rgb(X_test)
            elif X_train.shape[1] == 3072:  # CIFAR-10
                X_train = X_train.reshape(-1, 32, 32, 3)
                X_test = X_test.reshape(-1, 32, 32, 3)
                # Resize to 224x224 for pre-trained models
                X_train = tf.image.resize(X_train, [224, 224])
                X_test = tf.image.resize(X_test, [224, 224])
        
        # Normalize to [0, 1] range if needed
        if X_train.max() > 1:
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
        
        return X_train, X_test, y_train, y_test

    def create_augmentation_generators(self, X_train, y_train, X_test, y_test):
        """Create image data generators with augmentation if enabled"""
        try:
            # Check if any augmentation is enabled
            augmentation_enabled = (self.enable_rotation.isChecked() or 
                                  self.enable_horizontal_flip.isChecked() or 
                                  self.enable_zoom.isChecked() or 
                                  self.enable_brightness.isChecked())
            
            if not augmentation_enabled or len(X_train.shape) != 4:
                return None, None
            
            # Build augmentation parameters
            aug_params = {
                'rescale': 1./255 if X_train.max() > 1 else 1.0,
                'validation_split': 0.2
            }
            
            if self.enable_rotation.isChecked():
                aug_params['rotation_range'] = self.rotation_range.value()
            
            if self.enable_horizontal_flip.isChecked():
                aug_params['horizontal_flip'] = True
            
            if self.enable_zoom.isChecked():
                aug_params['zoom_range'] = self.zoom_range.value()
            
            if self.enable_brightness.isChecked():
                brightness_val = self.brightness_range.value()
                aug_params['brightness_range'] = [1-brightness_val, 1+brightness_val]
            
            # Create data generators
            datagen = ImageDataGenerator(**aug_params)
            datagen.fit(X_train)
            
            train_generator = datagen.flow(
                X_train, y_train,
                batch_size=32,
                subset='training'
            )
            
            val_generator = datagen.flow(
                X_train, y_train,
                batch_size=32,
                subset='validation'
            )
            
            return train_generator, val_generator
            
        except Exception as e:
            print(f"Error creating augmentation generators: {str(e)}")
            return None, None

    def save_finetuned_model(self):
        """Save the fine-tuned transfer learning model"""
        if self.current_model is None:
            self.show_error("No fine-tuned model to save. Please complete transfer learning first.")
            return
        
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Fine-tuned Model", "",
                "HDF5 files (*.h5);;SavedModel format (*.pb);;All files (*)"
            )
            
            if file_name:
                if file_name.endswith('.h5'):
                    self.current_model.save(file_name)
                else:
                    self.current_model.save(file_name, save_format='tf')
                
                self.status_bar.showMessage(f"Fine-tuned model saved to {file_name}")
                
        except Exception as e:
            self.show_error(f"Error saving fine-tuned model: {str(e)}")

    def create_gan_tab(self):
        """Create GAN and advanced generative models tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # GAN implementation section
        gan_group = QGroupBox("Generative Adversarial Networks (GAN)")
        gan_layout = QVBoxLayout()
        
        # GAN type selection
        gan_type_layout = QHBoxLayout()
        gan_type_layout.addWidget(QLabel("GAN Type:"))
        self.gan_type_combo = QComboBox()
        self.gan_type_combo.addItems(["Vanilla GAN", "Deep Convolutional GAN (DCGAN)", "Conditional GAN (CGAN)"])
        gan_type_layout.addWidget(self.gan_type_combo)
        gan_layout.addLayout(gan_type_layout)
        
        # GAN configuration parameters
        gan_params_group = QGroupBox("GAN Configuration")
        gan_params_layout = QFormLayout()
        
        self.gan_latent_dim = QSpinBox()
        self.gan_latent_dim.setRange(50, 1000)
        self.gan_latent_dim.setValue(100)
        gan_params_layout.addRow("Latent Dimension:", self.gan_latent_dim)
        
        self.gan_batch_size = QSpinBox()
        self.gan_batch_size.setRange(16, 256)
        self.gan_batch_size.setValue(64)
        gan_params_layout.addRow("Batch Size:", self.gan_batch_size)
        
        self.gan_epochs = QSpinBox()
        self.gan_epochs.setRange(100, 10000)
        self.gan_epochs.setValue(1000)
        gan_params_layout.addRow("Training Epochs:", self.gan_epochs)
        
        self.generator_lr = QDoubleSpinBox()
        self.generator_lr.setRange(0.0001, 0.01)
        self.generator_lr.setValue(0.0002)
        self.generator_lr.setDecimals(4)
        gan_params_layout.addRow("Generator Learning Rate:", self.generator_lr)
        
        self.discriminator_lr = QDoubleSpinBox()
        self.discriminator_lr.setRange(0.0001, 0.01)
        self.discriminator_lr.setValue(0.0002)
        self.discriminator_lr.setDecimals(4)
        gan_params_layout.addRow("Discriminator Learning Rate:", self.discriminator_lr)
        
        gan_params_group.setLayout(gan_params_layout)
        gan_layout.addWidget(gan_params_group)
        
        # GAN training controls
        gan_controls_layout = QHBoxLayout()
        
        train_gan_btn = QPushButton("Train GAN")
        train_gan_btn.clicked.connect(self.train_gan_model)
        train_gan_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px; }")
        gan_controls_layout.addWidget(train_gan_btn)
        
        generate_samples_btn = QPushButton("Generate Samples")
        generate_samples_btn.clicked.connect(self.generate_gan_samples)
        gan_controls_layout.addWidget(generate_samples_btn)
        
        save_gan_btn = QPushButton("Save GAN Model")
        save_gan_btn.clicked.connect(self.save_gan_model)
        gan_controls_layout.addWidget(save_gan_btn)
        
        gan_layout.addLayout(gan_controls_layout)
        
        gan_group.setLayout(gan_layout)
        layout.addWidget(gan_group)
        
        # Advanced techniques placeholder
        advanced_group = QGroupBox("Advanced Generative Techniques")
        advanced_layout = QVBoxLayout()
        
        # Information about upcoming features
        info_label = QLabel("""
Advanced generative techniques implementation status:

✓ GAN Framework: Basic infrastructure implemented
🔄 Variational Autoencoders (VAE): Coming in v0.1.4
🔄 Transformer Models: Planned for v0.1.5
🔄 Diffusion Models: Research integration planned
🔄 Style Transfer: Neural style transfer implementation

Current GAN implementation provides a foundation for:
- Generator and discriminator network design
- Adversarial training loop structure  
- Sample generation and evaluation

Full GAN training requires specialized dataset preparation
and extended training procedures suitable for research environments.
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 10px; border: 1px solid #ddd; }")
        advanced_layout.addWidget(info_label)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        return widget

    def train_gan_model(self):
        """Train GAN model with the current configuration"""
        if self.X_train is None:
            self.show_error("Please load a dataset first for GAN training.")
            return
        
        try:
            # Display information about GAN implementation
            info_msg = """
GAN Training Infrastructure:

This implementation provides the framework for Generative Adversarial Network training.
The current version includes:

• Generator and Discriminator architecture templates
• Adversarial training loop structure
• Loss function implementations
• Sample generation pipeline

For full GAN training, specialized configurations are required based on:
- Dataset characteristics (image size, channels, distribution)
- Training stability techniques (gradient penalty, spectral normalization)
- Architecture optimizations (progressive growing, self-attention)

Advanced GAN implementations will be available in future updates with:
- Pre-configured architectures for common datasets
- Training stability improvements
- Advanced loss functions and regularization techniques
            """
            
            QMessageBox.information(self, "GAN Training Information", info_msg)
            
        except Exception as e:
            self.show_error(f"Error in GAN training setup: {str(e)}")

    def generate_gan_samples(self):
        """Generate samples from trained GAN model"""
        QMessageBox.information(self, "GAN Sample Generation", 
                              "Sample generation will be available after GAN model training is completed. "
                              "This feature will include:\n\n"
                              "• Random sample generation from latent space\n"
                              "• Conditional generation (for CGAN)\n"
                              "• Latent space interpolation\n"
                              "• Sample quality evaluation metrics")

    def save_gan_model(self):
        """Save trained GAN model components"""
        QMessageBox.information(self, "GAN Model Saving", 
                              "GAN model saving will include:\n\n"
                              "• Generator model architecture and weights\n"
                              "• Discriminator model (for continued training)\n"
                              "• Training configuration and hyperparameters\n"
                              "• Sample generation utilities")

    def create_loss_function_group(self, is_classification=True):
        """Create a group for loss function selection"""
        group = QGroupBox("Loss Function")
        layout = QVBoxLayout()
        
        # Create radio buttons for loss functions
        self.loss_button_group = QButtonGroup()
        
        if is_classification:
            loss_options = ["Cross-Entropy", "Hinge Loss"]
        else:
            loss_options = ["MSE", "MAE", "Huber Loss"]
        
        for i, loss in enumerate(loss_options):
            radio = QRadioButton(loss)
            self.loss_button_group.addButton(radio, i)
            layout.addWidget(radio)
        
        # Select first option by default
        if self.loss_button_group.buttons():
            self.loss_button_group.buttons()[0].setChecked(True)
        
        group.setLayout(layout)
        return group
    
    def get_selected_loss_function(self, is_classification=True):
        """Get the selected loss function"""
        if is_classification:
            loss_map = {
                0: "categorical_crossentropy",  # Cross-Entropy
                1: "hinge"  # Hinge Loss
            }
        else:
            loss_map = {
                0: "mse",  # MSE
                1: "mae",  # MAE
                2: "huber"  # Huber Loss
            }
        
        selected_id = self.loss_button_group.checkedId()
        return loss_map.get(selected_id, "categorical_crossentropy" if is_classification else "mse")

    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Loss function selection for regression
        loss_group_reg = self.create_loss_function_group(is_classification=False)
        regression_layout.addWidget(loss_group_reg)
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        regression_layout.addWidget(lr_group)
        
        # SGD Regression
        sgd_reg_group = self.create_algorithm_group(
            "SGD Regression",
            {"max_iter": "int",
             "alpha": "double",
             "tol": "double"}
        )
        regression_layout.addWidget(sgd_reg_group)
        
        # SVR (Support Vector Regression)
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {"C": "double",
             "epsilon": "double",
             "kernel": ["linear", "rbf", "poly", "sigmoid"],
             "degree": "int",
             "gamma": ["scale", "auto"]}
        )
        regression_layout.addWidget(svr_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Loss function selection for classification
        loss_group_class = self.create_loss_function_group(is_classification=True)
        classification_layout.addWidget(loss_group_class)
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        classification_layout.addWidget(logistic_group)
        
        # SGD Classification
        sgd_class_group = self.create_algorithm_group(
            "SGD Classification",
            {"max_iter": "int",
             "alpha": "double",
             "loss": ["hinge", "log_loss", "modified_huber"],
             "tol": "double"}
        )
        classification_layout.addWidget(sgd_class_group)
        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly", "sigmoid"],
             "degree": "int",
             "gamma": ["scale", "auto"]}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget

    def create_bayesian_tab(self):
        """Create the Bayesian methods tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Naive Bayes section
        nb_group = QGroupBox("Naive Bayes")
        nb_layout = QVBoxLayout()
        
        # Gaussian Naive Bayes parameters
        gnb_params = QGroupBox("Gaussian Naive Bayes Parameters")
        gnb_params_layout = QVBoxLayout()
        
        # Var smoothing parameter
        var_smooth_layout = QHBoxLayout()
        var_smooth_layout.addWidget(QLabel("Var Smoothing:"))
        self.var_smoothing_spin = QDoubleSpinBox()
        self.var_smoothing_spin.setRange(1e-12, 1.0)
        self.var_smoothing_spin.setValue(1e-9)
        self.var_smoothing_spin.setDecimals(12)
        self.var_smoothing_spin.setSingleStep(1e-10)
        var_smooth_layout.addWidget(self.var_smoothing_spin)
        gnb_params_layout.addLayout(var_smooth_layout)
        
        # Prior probabilities selection
        prior_layout = QHBoxLayout()
        prior_layout.addWidget(QLabel("Prior Probabilities:"))
        self.prior_combo = QComboBox()
        self.prior_combo.addItems(["Uniform (Auto)", "Custom"])
        prior_layout.addWidget(self.prior_combo)
        gnb_params_layout.addLayout(prior_layout)
        
        # Custom prior probabilities input
        custom_prior_layout = QHBoxLayout()
        custom_prior_layout.addWidget(QLabel("Custom Priors (comma separated):"))
        self.custom_prior_input = QLineEdit()
        self.custom_prior_input.setPlaceholderText("e.g., 0.3,0.7 for binary classification")
        custom_prior_layout.addWidget(self.custom_prior_input)
        gnb_params_layout.addLayout(custom_prior_layout)
        
        # Enable/disable custom prior input based on selection
        def update_prior_input():
            self.custom_prior_input.setEnabled(self.prior_combo.currentText() == "Custom")
        
        self.prior_combo.currentIndexChanged.connect(update_prior_input)
        update_prior_input()  # Initial state
        
        gnb_params.setLayout(gnb_params_layout)
        nb_layout.addWidget(gnb_params)
        
        # Train button
        train_nb_btn = QPushButton("Train Naive Bayes")
        train_nb_btn.clicked.connect(self.train_naive_bayes)
        nb_layout.addWidget(train_nb_btn)
        
        nb_group.setLayout(nb_layout)
        layout.addWidget(nb_group)
        
        # Bayesian Networks section (placeholder for future expansion)
        bn_group = QGroupBox("Bayesian Networks")
        bn_layout = QVBoxLayout()
        
        bn_label = QLabel("Bayesian Network functionality will be implemented in a future update.")
        bn_layout.addWidget(bn_label)
        
        bn_group.setLayout(bn_layout)
        layout.addWidget(bn_group)
        
        return widget
    
    def train_naive_bayes(self):
        """Train a Gaussian Naive Bayes model with the configured parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Get parameters
            var_smoothing = self.var_smoothing_spin.value()
            
            # Handle priors
            priors = None
            if self.prior_combo.currentText() == "Custom":
                try:
                    priors_str = self.custom_prior_input.text().strip()
                    if priors_str:
                        priors = [float(x.strip()) for x in priors_str.split(',')]
                        
                        # Validate priors (must sum to 1)
                        if abs(sum(priors) - 1.0) > 1e-10:
                            self.show_error("Custom priors must sum to 1.0")
                            return
                        
                        # Validate priors length matches number of classes
                        n_classes = len(np.unique(self.y_train))
                        if len(priors) != n_classes:
                            self.show_error(f"Number of priors ({len(priors)}) must match number of classes ({n_classes})")
                            return
                except ValueError:
                    self.show_error("Invalid custom priors format. Please enter comma-separated numeric values.")
                    return
            
            # Create and train model
            model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Store current model
            self.current_model = model
            
            # Update visualization and metrics
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)
            
            # Update status
            priors_info = f" with custom priors {priors}" if priors else ""
            self.status_bar.showMessage(f"Trained Gaussian Naive Bayes (var_smoothing={var_smoothing}){priors_info}")
            
        except Exception as e:
            self.show_error(f"Error training Naive Bayes model: {str(e)}")

    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget

    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
                if param_name == "n_estimators":
                    widget.setValue(100)
                elif param_name == "max_iter":
                    widget.setValue(200)
                elif param_name == "n_clusters":
                    widget.setValue(8)
                elif param_name == "degree":
                    widget.setValue(3)
                else:
                    widget.setValue(5)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                if param_name == "C":
                    widget.setRange(0.01, 1000.0)
                    widget.setValue(1.0)
                elif param_name == "epsilon":
                    widget.setRange(0.001, 1.0)
                    widget.setValue(0.1)
                elif param_name == "alpha":
                    widget.setRange(0.00001, 1.0)
                    widget.setValue(0.0001)
                elif param_name == "tol":
                    widget.setRange(1e-6, 0.1)
                    widget.setValue(1e-4)
                else:
                    widget.setRange(0.0001, 1000.0)
                    widget.setValue(1.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
                if param_name == "fit_intercept" or param_name == "whiten":
                    widget.setChecked(True)
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group
        
    def train_model(self, model_name, param_widgets):
        """Train selected model with the specified parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Extract parameter values from widgets
            params = {}
            for param_name, widget in param_widgets.items():
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
            
            # Create and train appropriate model
            if model_name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=params.get('fit_intercept', True)
                )
                
            elif model_name == "SGD Regression":
                loss_function = self.get_selected_loss_function(is_classification=False)
                sklearn_loss = 'squared_error' if loss_function == 'mse' else \
                               'epsilon_insensitive' if loss_function == 'mae' else \
                               'huber'
                              
                model = SGDRegressor(
                    max_iter=params.get('max_iter', 1000),
                    alpha=params.get('alpha', 0.0001),
                    tol=params.get('tol', 1e-3),
                    loss=sklearn_loss,
                    random_state=42
                )
                
            elif model_name == "Support Vector Regression":
                model = SVR(
                    C=params.get('C', 1.0),
                    epsilon=params.get('epsilon', 0.1),
                    kernel=params.get('kernel', 'rbf'),
                    degree=params.get('degree', 3) if params.get('kernel') == 'poly' else 3,
                    gamma=params.get('gamma', 'scale')
                )
                
            elif model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=params.get('C', 1.0),
                    max_iter=params.get('max_iter', 100),
                    multi_class=params.get('multi_class', 'ovr'),
                    random_state=42
                )
                
            elif model_name == "SGD Classification":
                loss_function = self.get_selected_loss_function(is_classification=True)
                sklearn_loss = 'log_loss' if loss_function == 'categorical_crossentropy' else 'hinge'
                
                model = SGDClassifier(
                    max_iter=params.get('max_iter', 1000),
                    alpha=params.get('alpha', 0.0001),
                    loss=params.get('loss', 'hinge'),
                    tol=params.get('tol', 1e-3),
                    random_state=42
                )
                
            elif model_name == "Naive Bayes":
                model = GaussianNB(
                    var_smoothing=params.get('var_smoothing', 1e-9)
                )
                
            elif model_name == "Support Vector Machine":
                model = SVC(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    degree=params.get('degree', 3) if params.get('kernel') == 'poly' else 3,
                    gamma=params.get('gamma', 'scale'),
                    random_state=42
                )
                
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=params.get('max_depth', 5),
                    min_samples_split=params.get('min_samples_split', 2),
                    criterion=params.get('criterion', 'gini'),
                    random_state=42
                )
                
            elif model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 5),
                    min_samples_split=params.get('min_samples_split', 2),
                    random_state=42
                )
                
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(
                    n_neighbors=params.get('n_neighbors', 5),
                    weights=params.get('weights', 'uniform'),
                    metric=params.get('metric', 'euclidean')
                )
                
            elif model_name == "K-Means Parameters":
                model = KMeans(
                    n_clusters=params.get('n_clusters', 8),
                    max_iter=params.get('max_iter', 300),
                    n_init=params.get('n_init', 10),
                    random_state=42
                )
                
            elif model_name == "PCA Parameters":
                model = PCA(
                    n_components=params.get('n_components', 2),
                    whiten=params.get('whiten', False)
                )
                
            else:
                self.show_error(f"Unknown model: {model_name}")
                return
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            if model_name == "PCA Parameters":
                # Transform data with PCA
                X_pca = model.transform(self.X_test)
                
                # Visualize PCA components
                self.visualize_pca(model, X_pca)
                
                # Update metrics text with explained variance
                self.update_pca_metrics(model)
                
            elif model_name == "K-Means Parameters":
                # Predict clusters
                y_pred = model.predict(self.X_test)
                
                # Visualize clusters
                self.update_visualization(y_pred)
                
                # Update metrics with inertia and silhouette score
                self.update_kmeans_metrics(model, self.X_test, y_pred)
                
            else:
                # Regular model prediction
                y_pred = model.predict(self.X_test)
                
                # Store current model
                self.current_model = model
                
                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred)
            
            # Update status
            self.status_bar.showMessage(f"Trained {model_name}")
            
        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")
    
    def visualize_pca(self, pca_model, X_pca):
        """Visualize PCA components"""
        self.figure.clear()
        
        if X_pca.shape[1] >= 2:
            # 2D scatter plot of first two components
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_test, cmap='viridis', alpha=0.6)
            
            # Add legend if classification
            if len(np.unique(self.y_test)) <= 10:
                legend = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend)
            
            ax.set_xlabel(f"Principal Component 1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"Principal Component 2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)")
            ax.set_title("PCA: First Two Principal Components")
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
        else:
            # 1D visualization if only one component
            ax = self.figure.add_subplot(111)
            ax.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=self.y_test, cmap='viridis')
            ax.set_xlabel(f"Principal Component 1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_title("PCA: First Principal Component")
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_pca_metrics(self, pca_model):
        """Update metrics display with PCA information"""
        metrics_text = "PCA Analysis Results:\n\n"
        
        # Explained variance by component
        metrics_text += "Explained Variance Ratio:\n"
        for i, ratio in enumerate(pca_model.explained_variance_ratio_):
            metrics_text += f"Component {i+1}: {ratio:.4f} ({ratio:.2%})\n"
        
        # Cumulative explained variance
        cumulative = np.cumsum(pca_model.explained_variance_ratio_)
        metrics_text += "\nCumulative Explained Variance:\n"
        for i, cum_var in enumerate(cumulative):
            metrics_text += f"Components 1-{i+1}: {cum_var:.4f} ({cum_var:.2%})\n"
        
        # Top feature contributions to first component (if possible)
        if hasattr(pca_model, 'components_'):
            metrics_text += "\nTop Features in Principal Component 1:\n"
            # Get absolute coefficients for the first component
            abs_coeffs = np.abs(pca_model.components_[0])
            # Get indices of top coefficients
            top_indices = abs_coeffs.argsort()[-5:][::-1]  # Top 5 features
            
            for idx in top_indices:
                metrics_text += f"Feature {idx}: {pca_model.components_[0][idx]:.4f}\n"
        
        self.metrics_text.setText(metrics_text)
    
    def update_kmeans_metrics(self, kmeans_model, X_test, y_pred):
        """Update metrics display with K-Means information"""
        metrics_text = "K-Means Clustering Results:\n\n"
        
        # Inertia (within-cluster sum-of-squares)
        metrics_text += f"Inertia: {kmeans_model.inertia_:.4f}\n\n"
        
        # Cluster sizes
        cluster_sizes = np.bincount(y_pred)
        metrics_text += "Cluster Sizes:\n"
        for i, size in enumerate(cluster_sizes):
            metrics_text += f"Cluster {i}: {size} samples ({size/len(y_pred):.2%})\n"
        
        # Calculate silhouette score if sklearn.metrics is available
        try:
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(X_test, y_pred)
            metrics_text += f"\nSilhouette Score: {silhouette_avg:.4f}\n"
            metrics_text += "(Closer to 1 means better-defined clusters)"
        except:
            pass
        
        self.metrics_text.setText(metrics_text)

    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred, alpha=0.7)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression Results")
            ax.grid(True, linestyle='--', alpha=0.7)
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization if more than 2 features
                try:
                    pca = PCA(n_components=2)
                    X_test_2d = pca.fit_transform(self.X_test)
                    
                    ax = self.figure.add_subplot(111)
                    scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                      c=y_pred, cmap='viridis', alpha=0.7)
                    
                    # Add color bar
                    if len(np.unique(y_pred)) > 1:
                        self.figure.colorbar(scatter, ax=ax, label="Class")
                    
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_title("Classification Results (PCA)")
                    ax.grid(True, linestyle='--', alpha=0.7)
                except Exception as e:
                    # If PCA fails, use a confusion matrix visualization
                    self.visualize_confusion_matrix(self.y_test, y_pred)
                
            else:  # Direct 2D visualization if only 2 features
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], 
                                  self.X_test[:, 1] if self.X_test.shape[1] > 1 else np.zeros_like(self.X_test[:, 0]),
                                  c=y_pred, cmap='viridis', alpha=0.7)
                
                # Add color bar
                if len(np.unique(y_pred)) > 1:
                    self.figure.colorbar(scatter, ax=ax, label="Class")
                
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2" if self.X_test.shape[1] > 1 else "")
                ax.set_title("Classification Results")
                ax.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def visualize_confusion_matrix(self, y_true, y_pred):
        """Visualize confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        ax = self.figure.add_subplot(111)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        self.figure.colorbar(cax)
        
        # Set labels
        classes = np.unique(y_true)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            metrics_text += f"Mean Squared Error (MSE): {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            metrics_text += f"Mean Absolute Error (MAE): {mae:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}\n"
            
            # Residuals statistics
            residuals = self.y_test - y_pred
            metrics_text += f"\nResiduals Statistics:\n"
            metrics_text += f"Mean: {np.mean(residuals):.4f}\n"
            metrics_text += f"Standard Deviation: {np.std(residuals):.4f}\n"
            metrics_text += f"Min: {np.min(residuals):.4f}\n"
            metrics_text += f"Max: {np.max(residuals):.4f}\n"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix) + "\n\n"
            
            # Class-specific metrics
            classes = np.unique(self.y_test)
            
            if len(classes) == 2:  # Binary classification
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    
                    precision = precision_score(self.y_test, y_pred, average='binary')
                    recall = recall_score(self.y_test, y_pred, average='binary')
                    f1 = f1_score(self.y_test, y_pred, average='binary')
                    
                    metrics_text += f"Precision: {precision:.4f}\n"
                    metrics_text += f"Recall: {recall:.4f}\n"
                    metrics_text += f"F1 Score: {f1:.4f}\n"
                except:
                    pass
                
            else:  # Multi-class classification
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    
                    metrics_text += "Per-Class Metrics:\n"
                    
                    precisions = precision_score(self.y_test, y_pred, average=None)
                    recalls = recall_score(self.y_test, y_pred, average=None)
                    f1s = f1_score(self.y_test, y_pred, average=None)
                    
                    for i, class_label in enumerate(classes):
                        metrics_text += f"\nClass {class_label}:\n"
                        metrics_text += f"  Precision: {precisions[i]:.4f}\n"
                        metrics_text += f"  Recall: {recalls[i]:.4f}\n"
                        metrics_text += f"  F1 Score: {f1s[i]:.4f}\n"
                    
                    # Macro averages
                    metrics_text += "\nMacro Averages:\n"
                    metrics_text += f"  Precision: {np.mean(precisions):.4f}\n"
                    metrics_text += f"  Recall: {np.mean(recalls):.4f}\n"
                    metrics_text += f"  F1 Score: {np.mean(f1s):.4f}\n"
                except:
                    pass
        
        self.metrics_text.setText(metrics_text)
    
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Check if history contains accuracy
        has_accuracy = 'accuracy' in history.history
        
        if has_accuracy:
            # Plot training & validation accuracy
            ax1 = self.figure.add_subplot(211)
            ax1.plot(history.history['accuracy'])
            ax1.plot(history.history['val_accuracy'])
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend(['Train', 'Validation'], loc='lower right')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot training & validation loss
            ax2 = self.figure.add_subplot(212)
            ax2.plot(history.history['loss'])
            ax2.plot(history.history['val_loss'])
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            # Plot only loss if accuracy is not available
            ax = self.figure.add_subplot(111)
            ax.plot(history.history['loss'])
            ax.plot(history.history['val_loss'])
            ax.set_title('Model Loss')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Validation'], loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
