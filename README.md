# Adeptus Vita

Adeptus Vita is an AI-powered platform for the diagnosis of Alzheimer's and dementia using MRI scans. The application provides a modern, responsive interface for uploading and analyzing brain scans, viewing diagnostic results, and accessing educational resources.

## Table of Contents

- [Features](#features)
- [Pages](#pages)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Technical Documentation: Improved ANN Model](#technical-documentation-improved-ann-model)
  - [Model Overview](#model-overview)
  - [Dataset](#dataset)
  - [Data Preparation Pipeline](#data-preparation-pipeline)
  - [Feature Extraction](#feature-extraction)
  - [Model Architecture and Optimization](#model-architecture-and-optimization)
  - [Performance Evaluation](#performance-evaluation)
  - [Inference Pipeline](#inference-pipeline)
  - [Model Usage Example](#model-usage-example)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality

- **AI-Powered Diagnostic Model**: Upload MRI scans to receive automated analysis for signs of neurodegenerative diseases
- **Diagnostic History**: View and track previous scan results and analyses
- **Educational Blog**: Access articles and research on Alzheimer's, dementia, and diagnostic technologies
- **User Accounts**: Secure user authentication and profile management

### Technical Features

- **Modern UI/UX**: Clean, responsive interface built with Next.js and Tailwind CSS
- **Shadcn UI Components**: Consistent design system with accessible components
- **Dark/Light Mode**: Theme switching with system preference detection
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Breadcrumb Navigation**: Clear navigation path throughout the application
- **Search Functionality**: Search across blog posts and diagnostic results
- **Notification System**: In-app notifications for diagnostic results and updates
- **Settings Management**: Comprehensive user preference controls

## Pages

- **Home**: Landing page with overview of the platform's capabilities
- **Diagnostic Model**: Upload and analyze MRI scans
- **Blog**: Educational articles and research updates
- **Search**: Search functionality across the application
- **Settings**: User account and application preferences
- **Notifications**: System and diagnostic notifications
- **About**: Information about the platform and team

## Getting Started

### Prerequisites

- Node.js 18.0 or later
- npm or yarn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dead-si1ence/adeptus-vita.git
   cd adeptus-vita
   ```

2. Install dependencies:

   ```bash
   npm install

   # or

   yarn install
   ```

3. Run the development server:

   ```bash
   npm run dev

   # or

   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Usage Guide

### Diagnostic Model

1. Navigate to the Diagnostic Model page
2. Upload an MRI scan in JPG, PNG, or DICOM format
3. Click "Analyze Scan" to process the image
4. View the diagnostic results, including confidence level and recommendations
5. Access your diagnostic history in the "Prediction History" tab

### Blog

1. Browse articles on the Blog page
2. Click on an article to read the full content
3. Navigate between articles using the "Continue Reading" section

### Search

1. Use the search bar in the header or the dedicated Search page
2. Enter keywords related to diagnostics, research, or blog content
3. Filter and sort results by type, date, or relevance

### Settings

1. Access the Settings page from the header menu
2. Customize appearance preferences (theme, font size)
3. Manage notification preferences
4. Update account information and security settings

### Notifications

1. View notifications by clicking the bell icon in the header
2. Mark notifications as read or delete them
3. Adjust notification preferences in the Settings page

## Project Structure

```plaintext
adeptus-vita/
├── app/                  # Next.js App Router pages
│   ├── about/            # About page
│   ├── blog/             # Blog pages
│   ├── model/            # Diagnostic model page
│   ├── notifications/    # Notifications page
│   ├── search/           # Search page
│   ├── settings/         # Settings page
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Home page
├── components/           # React components
│   ├── layout/           # Layout components
│   ├── ui/               # UI components (shadcn)
│   └── ...               # Other components
├── hooks/                # Custom React hooks
├── lib/                  # Utility functions
├── public/               # Static assets
└── ...                   # Configuration files
```

## Customization

### Themes

The application supports light and dark modes, with a system preference option. Theme settings can be adjusted in the Settings page or via the theme toggle in the header.

### UI Components

UI components are built using the shadcn/ui library, which provides a consistent design system. Components can be customized by modifying the corresponding files in the `components/ui` directory.

### Styling

Styling is implemented using Tailwind CSS. Global styles are defined in `app/globals.css`, and component-specific styles are applied using Tailwind utility classes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# Technical Documentation: Improved ANN Model

## Model Overview

The diagnostic model is an advanced Artificial Neural Network (ANN) designed to classify brain MRI images into three categories of Alzheimer's disease progression: Non-Demented, Mild Demented, and Moderate Demented. The model uses a hybrid approach combining transfer learning (with EfficientNetB0 for feature extraction) and an optimized ANN for classification.

## Dataset

- **Structure**: The dataset consists of brain MRI scans categorized into three classes:
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
  4. Features are split into training (70%), validation (15%), and test (15%) sets.

## Model Architecture and Optimization

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

### Training Configuration

- **Loss Function**: Sparse categorical cross-entropy
- **Metrics**: Accuracy
- **Epochs**: 20
- **Batch Size**: 32
- **Validation**: Performance monitored on validation set

## Performance Evaluation

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes
- **F1-score**: Weighted harmonic mean of precision and recall
- **AUC**: Area under the ROC curve (one-vs-rest approach)

The `MetricsReport` function provides comprehensive evaluation of the model's classification capabilities on the test set.

## Inference Pipeline

The model includes a complete inference pipeline for classifying new MRI images:

1. **Image Loading**: Loads a target image from a specified path
2. **Preprocessing**: Converts to grayscale, resizes, and normalizes the image
3. **Feature Extraction**: Uses the same EfficientNetB0 architecture to extract features
4. **Classification**: Applies the trained ANN to predict the dementia class
5. **Label Decoding**: Converts numeric predictions back to human-readable class names

The `PredictNewInstance` function encapsulates this entire workflow, making it easy to classify new, unseen images.

## Model Usage Example

Here's how to use the model to predict the class of a new MRI image:

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

### Technical Requirements

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
