# Training Instructions for HSI-FastSpectralNet Model

This document provides step-by-step instructions on how to train the `newFastViT` model on the Pavia University hyperspectral dataset using the files in the `letest` folder.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Directory Structure](#directory-structure)
3. [Dataset Setup](#dataset-setup)
4. [Environment Setup](#environment-setup)
5. [Training the Model](#training-the-model)
6. [Understanding the Training Process](#understanding-the-training-process)
7. [Model Configuration](#model-configuration)
8. [Results and Outputs](#results-and-outputs)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.8 or later** installed
- **CUDA-capable GPU** (recommended) or CPU for training
- **Pavia University dataset** files:
  - `PaviaU.mat` - Hyperspectral image data
  - `PaviaU_gt.mat` - Ground truth labels

---

## Directory Structure

The `letest` folder contains the following key files:

```
letest/
├── train_new_model.ipynb    # Main training notebook
├── model.py                  # Model architecture (newFastViT)
├── data_loader.py           # Data loading and preprocessing functions
├── utils.py                 # Utility functions (metrics, performance calculations)
├── requirements.txt         # Python dependencies
└── dataset/                  # Dataset folder (create this)
    ├── PaviaU.mat          # Hyperspectral image data
    └── PaviaU_gt.mat       # Ground truth labels
```

---

## Dataset Setup

1. **Create the dataset folder**:
   ```bash
   cd letest
   mkdir -p dataset
   ```

2. **Download the Pavia University dataset**:
   - Visit: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
   - Download the Pavia University dataset
   - Extract the following files:
     - `PaviaU.mat` → Place in `letest/dataset/PaviaU.mat`
     - `PaviaU_gt.mat` → Place in `letest/dataset/PaviaU_gt.mat`

3. **Verify dataset files**:
   ```bash
   ls -lh dataset/
   ```
   You should see both `.mat` files listed.

---

## Environment Setup

1. **Navigate to the letest folder**:
   ```bash
   cd letest
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install numpy matplotlib scipy torch torchvision transformers scikit-learn seaborn einops thop jupyter ipykernel
   ```

   Or install from requirements.txt and add missing packages:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision transformers scikit-learn seaborn einops thop jupyter ipykernel
   ```

4. **Verify installation**:
   ```python
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Training the Model

### Option 1: Using Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the training notebook**:
   - Navigate to `letest/train_new_model.ipynb`
   - Click to open it

3. **Update dataset paths** (Cell 3):
   ```python
   # Update these paths to point to your dataset files
   image_file = "./dataset/PaviaU.mat"      # Update this path
   gt_file = "./dataset/PaviaU_gt.mat"     # Update this path
   ```

4. **Run all cells sequentially**:
   - Execute each cell in order (Shift + Enter)
   - The notebook will:
     - Load and preprocess the data
     - Initialize the model
     - Train the model
     - Evaluate performance
     - Generate visualizations

### Option 2: Using Python Script

If you prefer to run training as a script, you can convert the notebook cells into a Python script or use the provided `train.py` file.

---

## Understanding the Training Process

The training process follows these steps:

### Step 1: Data Loading
- **File**: `data_loader.py` → `load_pavia_university()`
- Loads hyperspectral image data and ground truth labels from `.mat` files
- Dataset dimensions: 610×340 pixels, 103 spectral bands, 9 classes

### Step 2: Data Preprocessing
- **File**: `data_loader.py` → `preprocess_data()`
- Normalizes pixel values to [0, 1]
- Extracts 5×5 spatial patches around each pixel
- Creates spatial-spectral feature vectors
- Filters out background pixels (class 0)

### Step 3: Dataset Splitting
- Splits data into 80% training and 20% testing sets
- Uses stratified sampling to maintain class distribution
- Creates PyTorch Dataset objects for training and testing

### Step 4: Model Initialization
- **File**: `model.py` → `newFastViT`
- Initializes the Fast Vision Transformer architecture
- Model components:
  - **Patch Embedding**: Converts spatial-spectral patches to embeddings
  - **Efficient Attention**: Optimized attention mechanism
  - **Spectral Attention**: Specialized attention for spectral bands
  - **Transformer Blocks**: 4 layers with residual connections
  - **Classification Head**: Outputs class predictions

### Step 5: Training Configuration
- Uses HuggingFace `Trainer` API
- Training parameters:
  - Epochs: 20
  - Batch size: 32 (training), 64 (evaluation)
  - Learning rate: Auto-configured with warmup
  - Weight decay: 0.01
  - Evaluation: After each epoch

### Step 6: Model Training
- Trains the model on the training dataset
- Saves checkpoints after each epoch
- Keeps the best 3 models based on validation loss
- Output directory: `./results_new_model/`

### Step 7: Evaluation
- Evaluates model on test dataset
- Calculates metrics:
  - Overall Accuracy (OA)
  - Average Accuracy (AA)
  - Kappa Coefficient
  - F1 Score, Precision, Recall
  - Per-class accuracies

### Step 8: Performance Metrics
- **File**: `utils.py`
- Calculates:
  - Latency per image (ms)
  - Throughput (samples/sec)
  - Model parameters (millions)
  - GFLOPs (computational complexity)

### Step 9: Visualizations
- Generates confusion matrix
- Plots per-class accuracy
- Displays classification report

---

## Model Configuration

The model uses the following default configuration (can be modified in Cell 9):

```python
# Model hyperparameters
window_size = 5              # Spatial patch size (5×5)
patch_size = 4              # Patch embedding size
num_channels = 103           # Number of spectral bands
num_classes = 9              # Number of land-cover classes
embed_dim = 192              # Embedding dimension (must be divisible by num_heads)
num_heads = 4                # Number of attention heads
depth = 4                    # Number of transformer blocks
mlp_ratio = 4.0              # MLP expansion ratio
```

**Important**: `embed_dim` must be divisible by `num_heads`. The default configuration (192 / 4 = 48) satisfies this requirement.

---

## Results and Outputs

After training completes, you will find:

1. **Model Checkpoints**:
   - Location: `./results_new_model/checkpoint-*/`
   - Contains: Model weights, optimizer state, training state

2. **Training Logs**:
   - Console output with training progress
   - Evaluation metrics after each epoch

3. **Performance Metrics**:
   - Overall Accuracy, Average Accuracy, Kappa Coefficient
   - F1 Score, Precision, Recall
   - Latency, Throughput, Model Size, GFLOPs

4. **Visualizations**:
   - Confusion matrix heatmap
   - Per-class accuracy bar chart
   - Classification report

---

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**:
   - Reduce batch size: Change `per_device_train_batch_size` from 32 to 16 or 8
   - Reduce model size: Decrease `embed_dim` or `depth`

2. **Dataset Not Found**:
   - Verify dataset files are in `letest/dataset/` folder
   - Check file paths in Cell 3 of the notebook

3. **Import Errors**:
   - Ensure all dependencies are installed: `pip install torch transformers scikit-learn einops thop`
   - Verify you're in the correct directory (letest folder)

4. **Model Configuration Error**:
   - Ensure `embed_dim` is divisible by `num_heads`
   - Check that `window_size >= patch_size`

5. **Slow Training**:
   - Enable GPU acceleration: Verify CUDA is available
   - Reduce number of epochs for testing
   - Use mixed precision training (if supported)

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all file paths are correct
3. Ensure all dependencies are installed
4. Check that dataset files are properly formatted

---

## Quick Start Summary

```bash
# 1. Navigate to letest folder
cd letest

# 2. Create dataset folder and add your data
mkdir -p dataset
# Copy PaviaU.mat and PaviaU_gt.mat to dataset/

# 3. Install dependencies
pip install torch torchvision transformers scikit-learn seaborn einops thop jupyter

# 4. Start Jupyter Notebook
jupyter notebook

# 5. Open train_new_model.ipynb and:
#    - Update dataset paths in Cell 3
#    - Run all cells sequentially
```

---

## Additional Notes

- **Training Time**: Expect 1-3 hours on GPU, 5-10 hours on CPU (depending on hardware)
- **Model Size**: Approximately 0.5-2 MB depending on configuration
- **Memory Requirements**: ~4-8 GB GPU memory recommended
- **Dataset Size**: Pavia University dataset is ~50 MB

---

## Model Architecture Details

The `newFastViT` model architecture:

- **Input**: 5×5×103 spatial-spectral patches
- **Patch Embedding**: Convolutional layer converts patches to embeddings
- **Positional Encoding**: Learnable positional embeddings
- **Transformer Blocks**: 4 layers with:
  - Efficient Attention mechanism
  - Layer Normalization
  - MLP with GELU activation
  - Residual connections
- **Spectral Attention**: Processes spectral information
- **Classification Head**: Linear layer outputs class probabilities

---

**Last Updated**: 2025
**Author**: Training Instructions for HSI-FastSpectralNet

