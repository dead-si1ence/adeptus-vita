# Technical Documentation: Improved ANN Model for Alzheimer's Disease Classification

## Overview

This document provides technical details for an advanced Artificial Neural Network (ANN) model designed to classify brain MRI images into three categories of Alzheimer's disease progression: Non-Demented, Mild Demented, and Moderate Demented. The model uses a hybrid approach combining transfer learning (with EfficientNetB0 for feature extraction) and an optimized ANN for classification.

## Table of Contents

- [Dataset](#dataset)
- [Data Preparation Pipeline](#data-preparation-pipeline)
  - [Data Rebalancing](#data-rebalancing)
  - [Image Preprocessing](#image-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Dataset Splitting](#dataset-splitting)
- [Model Architecture and Hyperparameter Optimization](#model-architecture-and-hyperparameter-optimization)
- [Model Training](#model-training)
- [Performance Evaluation](#performance-evaluation)
- [Inference Pipeline](#inference-pipeline)
- [Model Persistence](#model-persistence)
- [Technical Requirements](#technical-requirements)
- [Usage Example](#usage-example)
- [Potential Improvements](#potential-improvements)

## Dataset

- **Source**: The dataset consists of brain MRI scans categorized into three classes:
  - Non-Demented
  - Mild Demented
  - Moderate Demented

- **Class Imbalance**: The original dataset is imbalanced, with varying numbers of images per class.
- **Resolution**: The images are standardized to 300×300 pixels during preprocessing.

## Data Preparation Pipeline

### Data Rebalancing

The dataset undergoes a two-step rebalancing process to ensure equal representation of all classes:

#### Undersampling

- **Purpose**: Reduce the majority class (Mild Demented) to match the average number of images across classes.
- **Implementation**: The `UnderSampler` function randomly selects a subset of images from the majority class.
- **Algorithm**:
  1. Calculate the average number of images across all classes.
  2. Determine the number of images to remove from the majority class.
  3. Randomly select images to keep using `numpy.random.choice`.
  4. Copy selected images to a new directory.

#### Oversampling

- **Purpose**: Increase the minority classes (Moderate Demented, Non-Demented) to match the average class size.
- **Implementation**: The `OverSampler` function uses data augmentation to generate synthetic images.
- **Augmentation Parameters**:
  - Rotation range: 15 degrees
  - Width shift range: 7%
  - Height shift range: 7%
  - Shear range: 20%
  - Zoom range: 15%
  - Horizontal flip: Enabled
  - Fill mode: "nearest"
- **Algorithm**:
  1. Calculate the number of images to generate for each minority class.
  2. Randomly select source images for augmentation.
  3. Apply the image data generator to create augmented copies.
  4. Copy both original and generated images to new directories.

### Image Preprocessing

All images undergo a standardized preprocessing workflow:

- **Grayscale Conversion**: Converts RGB images to grayscale to reduce dimensionality.
- **Resizing**: Standardizes all images to 300×300 pixels.
- **Channel Conversion**: Replicates the grayscale channel across three channels for compatibility with EfficientNetB0.
- **Normalization**: Applies EfficientNet-specific preprocessing for optimal feature extraction.

## Feature Extraction

- **Model**: EfficientNetB0 pre-trained on ImageNet
- **Configuration**:
  - Input shape: (300, 300, 3)
  - Top layer: Excluded (include_top=False)
  - Pooling: Global average (pooling='avg')
  - Weights: ImageNet pre-trained

- **Process**:
  1. Images are processed in batches (default: 32, configurable).
  2. TensorFlow Dataset API is used for efficient data loading and processing.
  3. Feature vectors are extracted from the global average pooling layer.

## Dataset Splitting

The extracted features are split using a stratified approach:

- Training set: 70% of the data
- Validation set: 15% of the data
- Test set: 15% of the data

The stratification ensures that class distributions are maintained across all splits.

## Model Architecture and Hyperparameter Optimization

### Optimization Framework

- **Tool**: Optuna - a hyperparameter optimization framework
- **Objective**: Maximize validation accuracy
- **Trials**: 15 optimization trials

### Hyperparameters Searched

- **Activation Function**: ["relu", "sigmoid", "tanh"]
- **Optimizer**: ["adam", "sgd", "rmsprop"]
- **Learning Rate**: Range from 1e-5 to 1e-2 (log scale)
- **Dropout Rate**: Range from 0.1 to 0.5
- **Number of Layers**: 1 to 3 dense layers
- **Units per Layer**: [64, 128, 256]

### Final Model Architecture

The optimal architecture is determined by Optuna and used to build the final model. The architecture consists of:

1. Input layer matching the feature dimensionality from EfficientNetB0
2. Multiple dense layers with the optimal activation function
3. Dropout layers after each dense layer to prevent overfitting
4. Output layer with softmax activation for 3-class classification

## Model Training

- **Loss Function**: Sparse categorical cross-entropy
- **Optimization**: Selected by hyperparameter search (Adam, SGD, or RMSprop)
- **Metrics**: Accuracy
- **Epochs**: 20
- **Batch Size**: 32
- **Monitoring**: Validation accuracy tracked during training

## Performance Evaluation

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes
- **F1-score**: Weighted harmonic mean of precision and recall
- **AUC**: Area under the ROC curve (one-vs-rest approach)

The model's performance metrics are calculated using the `MetricsReport` function, which provides a comprehensive evaluation of the model's classification capabilities on the test set.

## Inference Pipeline

The model includes a complete inference pipeline for classifying new MRI images:

1. **Image Loading**: Loads a target image from a specified path
2. **Preprocessing**: Converts to grayscale, resizes, and normalizes the image
3. **Feature Extraction**: Uses the same EfficientNetB0 architecture to extract features
4. **Classification**: Applies the trained ANN to predict the dementia class
5. **Label Decoding**: Converts numeric predictions back to human-readable class names

The `PredictNewInstance` function encapsulates this entire workflow, making it easy to classify new, unseen images.

## Model Persistence

The trained model and associated components are saved for future use:

- **ANN Model**: Stored in both .keras and .h5 formats
- **Label Encoder**: Saved as a pickle file for consistent class mapping
- **Features and Labels**: Saved as NumPy arrays for potential retraining or analysis

## Technical Requirements

- **Libraries**:
  - TensorFlow/Keras for deep learning
  - scikit-learn for data splitting and evaluation metrics
  - Optuna for hyperparameter optimization
  - NumPy for numerical operations
  - PIL for image processing
  - Matplotlib for visualization
  - Pickle for serialization

- **Hardware Recommendations**:
  - GPU acceleration for model training and feature extraction
  - Sufficient RAM for processing the dataset batches

## Usage Example

The notebook demonstrates the full workflow from data preparation to model training, evaluation, and inference. The final section shows how to:

1. Load the saved model from disk
2. Process a new MRI image
3. Extract features using EfficientNetB0
4. Generate a prediction using the ANN model
5. Convert the numeric prediction to a human-readable diagnosis

```python
from tensorflow.keras.models import load_model
import pickle

# Load the model from disk
model_path = "Models/ANN.h5"
ANN_model = load_model(model_path)

# Load the label encoder
with open('Models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict on a new image
img_path = "path/to/new/mri_image.jpg"
predicted_label = PredictNewInstance(img_path, ANN_model, showImage=True)
print(f"Predicted diagnosis: {predicted_label}")
```

## Potential Improvements

Areas for potential enhancement in future iterations:

- Additional data augmentation techniques
- Experimentation with other pre-trained CNN architectures
- Ensemble methods combining multiple classifiers
- Attention mechanisms to focus on diagnostically relevant image regions
- Integration of additional clinical data for multimodal classification
- Explainable AI techniques to provide insights into model decisions
- Fine-tuning the EfficientNetB0 model for domain-specific feature extraction
