import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt
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
from tensorflow.keras import layers, models, optimizers, losses

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
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
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
        
        # Default imputer
        self.imputer = None
        
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
            "MNIST Dataset"
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
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Bayesian Methods", self.create_bayesian_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
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
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Loss function selection for classification
        loss_group_class = self.create_loss_function_group(is_classification=True)
        classification_layout.addWidget(loss_group_class)
        
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
    
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Loss function selection for neural networks
        loss_group_nn = self.create_loss_function_group()
        mlp_layout.addWidget(loss_group_nn)
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Current layers display
        self.layers_display = QTextEdit()
        self.layers_display.setReadOnly(True)
        self.layers_display.setMaximumHeight(150)
        self.layers_display.setText("No layers added yet.")
        mlp_layout.addWidget(self.layers_display)
        
        # Reset layers button
        reset_layers_btn = QPushButton("Reset Layers")
        reset_layers_btn.clicked.connect(self.reset_layers)
        mlp_layout.addWidget(reset_layers_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def reset_layers(self):
        """Reset the neural network layer configuration"""
        self.layer_config = []
        self.layers_display.setText("No layers added yet.")
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax", "linear"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "linear"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "MaxPooling2D":
                pool_size_label = QLabel("Pool Size:")
                pool_size_input = QLineEdit()
                pool_size_input.setText("2, 2")
                self.layer_param_inputs["pool_size"] = pool_size_input
                
                params_layout.addWidget(pool_size_label)
                params_layout.addWidget(pool_size_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name in ["kernel_size", "pool_size"]:
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            # Update layers display
            self.update_layers_display()
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def update_layers_display(self):
        """Update the text display showing current neural network layers"""
        if not self.layer_config:
            self.layers_display.setText("No layers added yet.")
            return
        
        layers_text = "Current Network Architecture:\n\n"
        
        for i, layer in enumerate(self.layer_config):
            layer_type = layer["type"]
            params = layer["params"]
            
            layers_text += f"{i+1}. {layer_type}: "
            
            if layer_type == "Dense":
                layers_text += f"{params.get('units')} units, {params.get('activation')} activation"
            elif layer_type == "Conv2D":
                layers_text += f"{params.get('filters')} filters, {params.get('kernel_size')} kernel, {params.get('activation')} activation"
            elif layer_type == "MaxPooling2D":
                layers_text += f"pool_size={params.get('pool_size')}"
            elif layer_type == "Dropout":
                layers_text += f"rate={params.get('rate')}"
            elif layer_type == "Flatten":
                layers_text += "flatten data"
            
            layers_text += "\n"
        
        self.layers_display.setText(layers_text)
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Define a standard CNN architecture with preset options
        cnn_layout = QVBoxLayout()
        
        # Image input shape
        input_shape_layout = QHBoxLayout()
        input_shape_layout.addWidget(QLabel("Input Shape (height, width, channels):"))
        self.cnn_input_height = QSpinBox()
        self.cnn_input_height.setRange(16, 512)
        self.cnn_input_height.setValue(28)
        self.cnn_input_width = QSpinBox()
        self.cnn_input_width.setRange(16, 512)
        self.cnn_input_width.setValue(28)
        self.cnn_input_channels = QSpinBox()
        self.cnn_input_channels.setRange(1, 4)
        self.cnn_input_channels.setValue(1)
        
        input_shape_layout.addWidget(self.cnn_input_height)
        input_shape_layout.addWidget(self.cnn_input_width)
        input_shape_layout.addWidget(self.cnn_input_channels)
        cnn_layout.addLayout(input_shape_layout)
        
        # Number of convolutional blocks
        blocks_layout = QHBoxLayout()
        blocks_layout.addWidget(QLabel("Number of Conv Blocks:"))
        self.cnn_blocks = QSpinBox()
        self.cnn_blocks.setRange(1, 5)
        self.cnn_blocks.setValue(2)
        blocks_layout.addWidget(self.cnn_blocks)
        cnn_layout.addLayout(blocks_layout)
        
        # Filters per block
        filters_layout = QHBoxLayout()
        filters_layout.addWidget(QLabel("Starting Filters:"))
        self.cnn_filters = QSpinBox()
        self.cnn_filters.setRange(8, 256)
        self.cnn_filters.setValue(32)
        filters_layout.addWidget(self.cnn_filters)
        cnn_layout.addLayout(filters_layout)
        
        # Dense layers at the end
        dense_layout = QHBoxLayout()
        dense_layout.addWidget(QLabel("Dense Layer Units:"))
        self.cnn_dense = QSpinBox()
        self.cnn_dense.setRange(16, 1024)
        self.cnn_dense.setValue(128)
        dense_layout.addWidget(self.cnn_dense)
        cnn_layout.addLayout(dense_layout)
        
        # CNN Train Button
        self.cnn_train_btn = QPushButton("Train CNN")
        self.cnn_train_btn.clicked.connect(self.train_cnn)
        cnn_layout.addWidget(self.cnn_train_btn)
        
        layout.addLayout(cnn_layout)
        group.setLayout(layout)
        return group
    
    def train_cnn(self):
        """Train a Convolutional Neural Network with the configured parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        # Check if the dataset is appropriate for CNN (image data)
        if len(self.X_train.shape) != 2:
            self.show_error("The loaded dataset is not in the right format for CNN. Please use image data.")
            return
        
        try:
            # Get parameters
            input_height = self.cnn_input_height.value()
            input_width = self.cnn_input_width.value()
            input_channels = self.cnn_input_channels.value()
            num_blocks = self.cnn_blocks.value()
            starting_filters = self.cnn_filters.value()
            dense_units = self.cnn_dense.value()
            
            # Create CNN model
            model = models.Sequential()
            
            # Reshape input data if needed
            if len(self.X_train.shape) == 2:
                # Assuming we're using MNIST or similar where features are flattened
                # Try to reshape to square images with the specified channels
                pixels = int(np.sqrt(self.X_train.shape[1] / input_channels))
                
                if pixels * pixels * input_channels != self.X_train.shape[1]:
                    self.show_error(f"Cannot reshape input data to {input_height}x{input_width}x{input_channels}. " +
                                   f"Input shape: {self.X_train.shape}")
                    return
                
                # Reshape the data
                X_train_reshaped = self.X_train.reshape(-1, pixels, pixels, input_channels)
                X_test_reshaped = self.X_test.reshape(-1, pixels, pixels, input_channels)
            else:
                # Data is already in the right shape
                X_train_reshaped = self.X_train
                X_test_reshaped = self.X_test
            
            # Input layer
            model.add(layers.InputLayer(input_shape=(pixels, pixels, input_channels)))
            
            # Add convolutional blocks
            filters = starting_filters
            for i in range(num_blocks):
                # Convolutional layer
                model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
                # Another conv layer in the same block
                model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
                # Max pooling
                model.add(layers.MaxPooling2D((2, 2)))
                # Double the filters for the next block
                filters *= 2
            
            # Flatten and dense layers
            model.add(layers.Flatten())
            model.add(layers.Dense(dense_units, activation='relu'))
            model.add(layers.Dropout(0.5))
            
            # Output layer
            num_classes = len(np.unique(self.y_train))
            model.add(layers.Dense(num_classes, activation='softmax'))
            
            # Get selected loss function
            loss_function = self.get_selected_loss_function(is_classification=True)
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.lr_spin.value()),
                loss=loss_function,
                metrics=['accuracy']
            )
            
            # Convert target to one-hot encoding if needed
            if loss_function == 'categorical_crossentropy':
                y_train_encoded = tf.keras.utils.to_categorical(self.y_train, num_classes)
                y_test_encoded = tf.keras.utils.to_categorical(self.y_test, num_classes)
            else:
                y_train_encoded = self.y_train
                y_test_encoded = self.y_test
            
            # Train model
            history = model.fit(
                X_train_reshaped, 
                y_train_encoded,
                batch_size=self.batch_size_spin.value(),
                epochs=self.epochs_spin.value(),
                validation_data=(X_test_reshaped, y_test_encoded),
                callbacks=[self.create_progress_callback()]
            )
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            # Make predictions
            y_pred = model.predict(X_test_reshaped)
            if loss_function == 'categorical_crossentropy':
                y_pred = np.argmax(y_pred, axis=1)
            
            # Update metrics
            self.update_metrics(y_pred)
            
            # Store current model
            self.current_model = model
            
            self.status_bar.showMessage("CNN Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training CNN: {str(e)}")
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # RNN Type
        rnn_type_layout = QHBoxLayout()
        rnn_type_layout.addWidget(QLabel("RNN Type:"))
        self.rnn_type_combo = QComboBox()
        self.rnn_type_combo.addItems(["SimpleRNN", "LSTM", "GRU"])
        rnn_type_layout.addWidget(self.rnn_type_combo)
        layout.addLayout(rnn_type_layout)
        
        # Sequence Length
        seq_length_layout = QHBoxLayout()
        seq_length_layout.addWidget(QLabel("Sequence Length:"))
        self.seq_length_spin = QSpinBox()
        self.seq_length_spin.setRange(1, 100)
        self.seq_length_spin.setValue(10)
        seq_length_layout.addWidget(self.seq_length_spin)
        layout.addLayout(seq_length_layout)
        
        # Number of RNN layers
        num_layers_layout = QHBoxLayout()
        num_layers_layout.addWidget(QLabel("Number of Layers:"))
        self.num_rnn_layers_spin = QSpinBox()
        self.num_rnn_layers_spin.setRange(1, 5)
        self.num_rnn_layers_spin.setValue(1)
        num_layers_layout.addWidget(self.num_rnn_layers_spin)
        layout.addLayout(num_layers_layout)
        
        # Units per layer
        units_layout = QHBoxLayout()
        units_layout.addWidget(QLabel("Units per Layer:"))
        self.rnn_units_spin = QSpinBox()
        self.rnn_units_spin.setRange(1, 512)
        self.rnn_units_spin.setValue(64)
        units_layout.addWidget(self.rnn_units_spin)
        layout.addLayout(units_layout)
        
        # Bidirectional checkbox
        bidirectional_layout = QHBoxLayout()
        bidirectional_layout.addWidget(QLabel("Bidirectional:"))
        self.bidirectional_check = QCheckBox()
        bidirectional_layout.addWidget(self.bidirectional_check)
        layout.addLayout(bidirectional_layout)
        
        # Placeholder message
        layout.addWidget(QLabel("RNN functionality will be fully implemented in a future update."))
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Get selected loss function
            loss_function = self.get_selected_loss_function()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # One-hot encode target for classification if using categorical_crossentropy
            num_classes = len(np.unique(self.y_train))
            if loss_function == 'categorical_crossentropy':
                y_train = tf.keras.utils.to_categorical(self.y_train, num_classes)
                y_test = tf.keras.utils.to_categorical(self.y_test, num_classes)
            else:
                y_train = self.y_train
                y_test = self.y_test
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss=loss_function,
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_test, y_test),
                              callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            # Make predictions
            y_pred = model.predict(X_test)
            if loss_function == 'categorical_crossentropy':
                y_pred = np.argmax(y_pred, axis=1)
            
            # Update metrics
            self.update_metrics(y_pred)
            
            # Store current model
            self.current_model = model
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        is_first_layer = True
        
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                if is_first_layer:
                    # Add input shape for the first layer
                    input_shape = self.X_train.shape[1:]
                    if len(input_shape) == 0:  # For 1D input
                        input_shape = (1,)
                    params = {**params, 'input_shape': input_shape}
                
                model.add(layers.Dense(**params))
                
            elif layer_type == "Conv2D":
                if is_first_layer:
                    # Add input shape for the first layer
                    if len(self.X_train.shape) <= 2:
                        # Need to reshape data for Conv2D
                        self.show_error("Data needs to be reshaped for Conv2D layer")
                        return None
                    params = {**params, 'input_shape': self.X_train.shape[1:]}
                
                model.add(layers.Conv2D(**params))
                
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D(pool_size=params.get('pool_size', (2, 2))))
                
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
                
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
            
            is_first_layer = False
        
        # If no output layer was added, add one based on the problem type
        if self.layer_config and self.layer_config[-1]["type"] != "Dense":
            num_classes = len(np.unique(self.y_train))
            
            # For regression or binary classification
            if num_classes <= 2:
                model.add(layers.Dense(1, activation='sigmoid' if num_classes == 2 else 'linear'))
            else:  # For multi-class classification
                model.add(layers.Dense(num_classes, activation='softmax'))
        
        return model
    
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_begin(self, epoch, logs=None):
                # Update progress at the start of each epoch
                progress = int((epoch / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
            def on_epoch_end(self, epoch, logs=None):
                # Update progress at the end of each epoch
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
    
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
