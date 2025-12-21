# Training Instructions for Houston Dataset

This document provides step-by-step instructions on how to train the `newFastViT` model on the **Houston hyperspectral dataset** using the files in the `letest` folder.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Houston Dataset Information](#houston-dataset-information)
3. [Prerequisites](#prerequisites)
4. [Step 1: Add Houston Dataset Support to data_loader.py](#step-1-add-houston-dataset-support-to-data_loaderpy)
5. [Step 2: Dataset Setup](#step-2-dataset-setup)
6. [Step 3: Modify Training Notebook](#step-3-modify-training-notebook)
7. [Step 4: Model Configuration for Houston](#step-4-model-configuration-for-houston)
8. [Step 5: Training the Model](#step-5-training-the-model)
9. [Differences from Pavia University](#differences-from-pavia-university)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Houston dataset is a hyperspectral image dataset commonly used for land-cover classification. To train the model on Houston dataset, you need to:

1. Add Houston dataset loading function to `data_loader.py`
2. Place Houston dataset files in the `dataset` folder
3. Modify the training notebook to use Houston dataset
4. Adjust model configuration for Houston dataset characteristics

---

## Houston Dataset Information

**Houston Dataset Characteristics:**
- **Dimensions**: Typically 349×1905 pixels (varies by version)
- **Spectral Bands**: 144 bands
- **Number of Classes**: 15 land-cover categories
- **File Format**: MATLAB `.mat` files
- **Variable Names**: Usually `'houston'` or `'Houston'` for image data, `'houston_gt'` or `'Houston_gt'` for ground truth

**Note**: The exact variable names in the `.mat` file may vary. You may need to inspect the file to determine the correct variable names.

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.8 or later** installed
- **CUDA-capable GPU** (recommended) or CPU for training
- **Houston dataset** files:
  - `Houston.mat` or `houston.mat` - Hyperspectral image data
  - `Houston_gt.mat` or `houston_gt.mat` - Ground truth labels
- All dependencies installed (same as Pavia University training)

---

## Step 1: Add Houston Dataset Support to data_loader.py

You need to add a function to load Houston dataset in `data_loader.py`. The preprocessing function can be reused, but you need a specific loader function.

### Option A: Add Function to Existing data_loader.py

Open `letest/data_loader.py` and add the following function after the `load_pavia_university` function:

```python
def load_houston(image_file, gt_file):
    """
    Load Houston hyperspectral dataset from .mat files.
    
    Args:
        image_file: Path to Houston image data .mat file
        gt_file: Path to Houston ground truth .mat file
    
    Returns:
        image_data: Hyperspectral image array (H, W, C)
        ground_truth: Ground truth labels array (H, W)
    """
    print("Loading Houston dataset...")
    
    # Load .mat files
    image_mat = scipy.io.loadmat(image_file)
    gt_mat = scipy.io.loadmat(gt_file)
    
    # Try common variable names for Houston dataset
    # Adjust these based on your actual .mat file structure
    possible_image_keys = ['houston', 'Houston', 'data', 'image']
    possible_gt_keys = ['houston_gt', 'Houston_gt', 'gt', 'ground_truth', 'map']
    
    # Find the correct key for image data
    image_data = None
    for key in possible_image_keys:
        if key in image_mat:
            image_data = image_mat[key]
            print(f"Found image data with key: '{key}'")
            break
    
    if image_data is None:
        print("Available keys in image file:", list(image_mat.keys()))
        raise ValueError("Could not find image data in .mat file. Please check variable names.")
    
    # Find the correct key for ground truth
    ground_truth = None
    for key in possible_gt_keys:
        if key in gt_mat:
            ground_truth = gt_mat[key]
            print(f"Found ground truth with key: '{key}'")
            break
    
    if ground_truth is None:
        print("Available keys in gt file:", list(gt_mat.keys()))
        raise ValueError("Could not find ground truth data in .mat file. Please check variable names.")
    
    # Remove MATLAB metadata keys (keys starting with '__')
    if image_data is None:
        image_keys = [k for k in image_mat.keys() if not k.startswith('__')]
        if len(image_keys) == 1:
            image_data = image_mat[image_keys[0]]
            print(f"Using image data key: '{image_keys[0]}'")
    
    if ground_truth is None:
        gt_keys = [k for k in gt_mat.keys() if not k.startswith('__')]
        if len(gt_keys) == 1:
            ground_truth = gt_mat[gt_keys[0]]
            print(f"Using ground truth key: '{gt_keys[0]}'")
    
    print(f"Image data shape: {image_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    
    return image_data, ground_truth
```

### Option B: Inspect Your .mat Files First

If you're unsure about the variable names, you can inspect the files first:

```python
import scipy.io

# Inspect image file
image_mat = scipy.io.loadmat('path/to/Houston.mat')
print("Keys in image file:", list(image_mat.keys()))

# Inspect ground truth file
gt_mat = scipy.io.loadmat('path/to/Houston_gt.mat')
print("Keys in gt file:", list(gt_mat.keys()))
```

Then use the actual variable names in the `load_houston` function.

---

## Step 2: Dataset Setup

1. **Create/Verify dataset folder**:
   ```bash
   cd letest
   mkdir -p dataset
   ```

2. **Download Houston Dataset**:
   - Download the Houston hyperspectral dataset from its official source
   - Common sources include IEEE GRSS Data Fusion Contest or research repositories
   - Ensure you have the required permissions to use the dataset

3. **Place dataset files**:
   ```
   letest/
   └── dataset/
       ├── Houston.mat      # Hyperspectral image data
       └── Houston_gt.mat   # Ground truth labels
   ```

4. **Verify files**:
   ```bash
   ls -lh dataset/
   ```

---

## Step 3: Modify Training Notebook

Open `letest/train_new_model.ipynb` and make the following modifications:

### Cell 1: Update Imports

Change the import statement to include Houston loader:

```python
# Import custom modules
from data_loader import load_houston, preprocess_data, PaviaUniversityDataset
# Note: PaviaUniversityDataset can be reused for Houston as it's generic
from model import newFastViT
from utils import (
    calculate_latency_per_image,
    calculate_throughput,
    overall_accuracy,
    average_accuracy,
    kappa_coefficient,
    calculate_f1_precision_recall,
    count_model_parameters,
    calculate_gflops
)
```

### Cell 0: Update Title and Description

```python
# Training New Model Architecture on Houston Dataset

This notebook trains the new model architecture from `model.py` on the Houston hyperspectral dataset.

## Model Architecture
- **newFastViT**: A novel Fast Vision Transformer with:
  - Efficient Attention mechanism
  - Spectral Attention module
  - Transformer blocks with residual connections
  - Optimized for hyperspectral image classification

## Dataset
- **Houston**: 349×1905 pixels, 144 spectral bands
- **Classes**: 15 land-cover categories
- **Window Size**: 5×5 spatial patches
```

### Cell 3: Update Dataset Paths and Loader

```python
# Update these paths to point to your Houston dataset files
image_file = "./dataset/Houston.mat"      # Update this path
gt_file = "./dataset/Houston_gt.mat"     # Update this path

# Load the Houston dataset
image_data, ground_truth = load_houston(image_file, gt_file)
print(f"\nDataset loaded successfully!")
print(f"Image data shape: {image_data.shape}")
print(f"Ground truth shape: {ground_truth.shape}")
```

---

## Step 4: Model Configuration for Houston

### Cell 9: Update Model Configuration

Houston dataset has different characteristics than Pavia University:

```python
# Initialize the model for Houston dataset
num_classes = len(np.unique(y))
num_channels = spatial_spectral_data.shape[-1]  # 144 spectral bands for Houston

# Model configuration for Houston dataset
window_size = 5
patch_size = 4
embed_dim = 192  # Must be divisible by num_heads (192 / 4 = 48)
num_heads = 4
depth = 4

# Validate configuration
if embed_dim % num_heads != 0:
    raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

# Calculate actual number of patches
actual_patches_h = (window_size - patch_size) // patch_size + 1
actual_patches_w = (window_size - patch_size) // patch_size + 1
actual_num_patches = actual_patches_h * actual_patches_w

print(f"Model Configuration for Houston Dataset:")
print(f"  Image size: {window_size}x{window_size}")
print(f"  Patch size: {patch_size}x{patch_size}")
print(f"  Actual patches: {actual_patches_h}x{actual_patches_w} = {actual_num_patches}")
print(f"  Number of spectral bands: {num_channels}")
print(f"  Number of classes: {num_classes}")
print(f"  Embed dim: {embed_dim} (divisible by num_heads={num_heads} ✓)")
print(f"  Head dim: {embed_dim // num_heads}")
print(f"  Depth: {depth}")

model = newFastViT(
    image_size=window_size,
    patch_size=patch_size,
    num_channels=num_channels,  # 144 for Houston
    num_classes=num_classes,     # 15 for Houston
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=4.0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"\nModel initialized successfully!")
print(f"Model device: {device}")
print(f"Number of parameters: {count_model_parameters(model):.2f} M")
```

---

## Step 5: Training the Model

1. **Open Jupyter Notebook**:
   ```bash
   cd letest
   jupyter notebook
   ```

2. **Open the modified notebook**: `train_new_model.ipynb`

3. **Run cells sequentially**:
   - Execute each cell in order (Shift + Enter)
   - The notebook will:
     - Load Houston dataset
     - Preprocess data (same preprocessing function works)
     - Initialize model with Houston-specific parameters
     - Train the model
     - Evaluate performance
     - Generate visualizations

4. **Monitor training**:
   - Watch for any errors related to dataset loading
   - Check that the number of classes matches (should be 15 for Houston)
   - Verify spectral band count is 144

---

## Differences from Pavia University

| Aspect | Pavia University | Houston Dataset |
|--------|------------------|-----------------|
| **Dimensions** | 610×340 pixels | 349×1905 pixels |
| **Spectral Bands** | 103 | 144 |
| **Number of Classes** | 9 | 15 |
| **Variable Names** | `'paviaU'`, `'paviaU_gt'` | `'houston'`, `'houston_gt'` (may vary) |
| **Model num_channels** | 103 | 144 |
| **Model num_classes** | 9 | 15 |

### Key Changes Required:

1. **Data Loader**: Use `load_houston()` instead of `load_pavia_university()`
2. **Model Configuration**: 
   - `num_channels=144` (instead of 103)
   - `num_classes=15` (instead of 9)
3. **Dataset Paths**: Point to Houston dataset files
4. **Preprocessing**: Same preprocessing function works (generic)

---

## Troubleshooting

### Issue 1: Variable Name Not Found

**Error**: `ValueError: Could not find image data in .mat file`

**Solution**:
1. Inspect the .mat file to find correct variable names:
   ```python
   import scipy.io
   mat = scipy.io.loadmat('dataset/Houston.mat')
   print(list(mat.keys()))
   ```
2. Update the `load_houston()` function with correct variable names
3. Or modify the function to automatically detect the data key

### Issue 2: Shape Mismatch

**Error**: Model expects different input dimensions

**Solution**:
- Verify `num_channels` matches actual spectral bands (144 for Houston)
- Check that preprocessing maintains correct dimensions
- Ensure `window_size` is appropriate (5×5 is standard)

### Issue 3: Number of Classes Mismatch

**Error**: Model output size doesn't match number of classes

**Solution**:
- Verify Houston dataset has 15 classes (check ground truth)
- Ensure `num_classes` in model initialization matches actual classes
- Check that label encoding is working correctly

### Issue 4: Memory Issues

**Error**: CUDA out of memory

**Solution**:
- Houston dataset is larger than Pavia, reduce batch size:
  ```python
  per_device_train_batch_size=16  # Instead of 32
  per_device_eval_batch_size=32   # Instead of 64
  ```
- Consider reducing model size (`embed_dim`, `depth`)
- Use gradient accumulation if needed

### Issue 5: Slow Training

**Solution**:
- Houston dataset is larger, training will take longer
- Ensure GPU is being used: `torch.cuda.is_available()` should be `True`
- Consider reducing number of epochs for initial testing
- Use mixed precision training if supported

---

## Quick Reference: Complete Modified Code Snippets

### Modified data_loader.py Addition

Add this function to `data_loader.py`:

```python
def load_houston(image_file, gt_file):
    """Load Houston hyperspectral dataset."""
    print("Loading Houston dataset...")
    image_mat = scipy.io.loadmat(image_file)
    gt_mat = scipy.io.loadmat(gt_file)
    
    # Auto-detect variable names (exclude MATLAB metadata)
    image_keys = [k for k in image_mat.keys() if not k.startswith('__')]
    gt_keys = [k for k in gt_mat.keys() if not k.startswith('__')]
    
    if len(image_keys) != 1:
        raise ValueError(f"Expected 1 data key in image file, found {len(image_keys)}: {image_keys}")
    if len(gt_keys) != 1:
        raise ValueError(f"Expected 1 data key in gt file, found {len(gt_keys)}: {gt_keys}")
    
    image_data = image_mat[image_keys[0]]
    ground_truth = gt_mat[gt_keys[0]]
    
    print(f"Image data shape: {image_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    
    return image_data, ground_truth
```

### Modified Notebook Cell 3

```python
# Houston dataset paths
image_file = "./dataset/Houston.mat"
gt_file = "./dataset/Houston_gt.mat"

# Load Houston dataset
image_data, ground_truth = load_houston(image_file, gt_file)
print(f"\nHouston dataset loaded successfully!")
print(f"Image data shape: {image_data.shape}")
print(f"Ground truth shape: {ground_truth.shape}")
```

### Modified Notebook Cell 9

```python
# Houston dataset configuration
num_classes = len(np.unique(y))
num_channels = spatial_spectral_data.shape[-1]  # Should be 144

model = newFastViT(
    image_size=5,
    patch_size=4,
    num_channels=num_channels,  # 144 for Houston
    num_classes=num_classes,     # 15 for Houston
    embed_dim=192,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0
)
```

---

## Summary Checklist

- [ ] Added `load_houston()` function to `data_loader.py`
- [ ] Downloaded Houston dataset files
- [ ] Placed dataset files in `letest/dataset/` folder
- [ ] Updated notebook Cell 0 (title and description)
- [ ] Updated notebook Cell 1 (imports - add `load_houston`)
- [ ] Updated notebook Cell 3 (dataset paths and loader function)
- [ ] Updated notebook Cell 9 (model configuration: num_channels=144, num_classes=15)
- [ ] Verified dataset loads correctly
- [ ] Verified model initializes with correct parameters
- [ ] Started training

---

**Last Updated**: 2025
**Author**: Training Instructions for Houston Dataset - HSI-FastSpectralNet

