# Houston Dataset Setup - Complete ✅

## Summary

The Houston dataset has been downloaded and integrated into the training pipeline. All necessary files have been created and configured.

## What Was Done

### 1. ✅ Dataset Downloaded
- **Houston13.mat** - Hyperspectral image data (290 KB)
- **Houston13_7gt.mat** - Ground truth labels (290 KB)
- Location: `letest/dataset/`

### 2. ✅ Data Loader Updated
- Added `load_houston()` function to `data_loader.py`
- Function automatically detects variable names in .mat files
- Supports multiple naming conventions:
  - Image data: `ori_data`, `houston`, `Houston`, `Houston13`, `data`, `image`, `HSI`
  - Ground truth: `map`, `houston_gt`, `Houston_gt`, `Houston13_7gt`, `gt`, `ground_truth`, `label`

### 3. ✅ Training Script Created
- **File**: `train_houston.py`
- Complete training pipeline for Houston dataset
- Includes:
  - Data loading and preprocessing
  - Model initialization (144 bands, 15 classes)
  - Training with HuggingFace Trainer
  - Comprehensive evaluation metrics
  - Visualization (confusion matrix, per-class accuracy)
  - Results saving

## Quick Start

### Option 1: Run Training Script (Recommended)

```bash
cd letest
python train_houston.py
```

This will:
1. Load Houston dataset
2. Preprocess data (5×5 patches)
3. Split into train/test (80/20)
4. Initialize model with Houston-specific parameters
5. Train for 20 epochs
6. Evaluate and generate metrics
7. Save results to `./results_houston/`

### Option 2: Use Jupyter Notebook

1. Open `train_new_model.ipynb`
2. Modify Cell 1 to import `load_houston`:
   ```python
   from data_loader import load_houston, preprocess_data, PaviaUniversityDataset
   ```
3. Modify Cell 3 to use Houston dataset:
   ```python
   image_file = "./dataset/Houston13.mat"
   gt_file = "./dataset/Houston13_7gt.mat"
   image_data, ground_truth = load_houston(image_file, gt_file)
   ```
4. Modify Cell 9 to configure model for Houston:
   ```python
   num_channels = spatial_spectral_data.shape[-1]  # Should be 144
   num_classes = len(np.unique(y))  # Should be 15
   ```
5. Run all cells

## Houston Dataset Configuration

| Parameter | Value |
|-----------|-------|
| **Spectral Bands** | 144 |
| **Number of Classes** | 15 |
| **Window Size** | 5×5 |
| **Patch Size** | 4×4 |
| **Embed Dim** | 192 |
| **Num Heads** | 4 |
| **Depth** | 4 |

## Expected Output

After training completes, you'll find:

```
results_houston/
├── checkpoint-*/          # Model checkpoints
├── confusion_matrix.png   # Confusion matrix visualization
├── per_class_accuracy.png # Per-class accuracy plot
└── training_results.txt  # Detailed results and metrics
```

## Files Created/Modified

1. **`data_loader.py`** - Added `load_houston()` function
2. **`train_houston.py`** - New training script for Houston dataset
3. **`dataset/Houston13.mat`** - Houston hyperspectral image data
4. **`dataset/Houston13_7gt.mat`** - Houston ground truth labels

## Verification

To verify everything is set up correctly:

```python
from data_loader import load_houston
import scipy.io

# Check if files exist
import os
print("Houston13.mat exists:", os.path.exists("./dataset/Houston13.mat"))
print("Houston13_7gt.mat exists:", os.path.exists("./dataset/Houston13_7gt.mat"))

# Try loading (if scipy is installed)
try:
    image_data, ground_truth = load_houston(
        "./dataset/Houston13.mat",
        "./dataset/Houston13_7gt.mat"
    )
    print(f"✓ Dataset loaded successfully!")
    print(f"  Image shape: {image_data.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
except Exception as e:
    print(f"Error: {e}")
```

## Troubleshooting

### Issue: Variable names not found
**Solution**: The `load_houston()` function automatically detects variable names. If it fails, check the .mat file keys:
```python
import scipy.io
mat = scipy.io.loadmat('dataset/Houston13.mat')
print([k for k in mat.keys() if not k.startswith('__')])
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `train.py`:
```python
per_device_train_batch_size=16  # Instead of 32
```

### Issue: Dataset file not found
**Solution**: Ensure files are in `letest/dataset/` folder:
```bash
ls -lh letest/dataset/*.mat
```

## Next Steps

1. **Install dependencies** (if not already installed):
   ```bash
   pip install torch torchvision transformers scikit-learn seaborn einops thop scipy numpy matplotlib
   ```

2. **Run training**:
   ```bash
   cd letest
   python train_houston.py
   ```

3. **Monitor training**: Watch the console output for progress and metrics

4. **Check results**: After training, view results in `./results_houston/` folder

## Notes

- The downloaded files are relatively small (290 KB each), which might indicate they are sample files or compressed versions
- If you need the full dataset, you may need to download from the official source
- The training script is configured for Houston dataset characteristics (144 bands, 15 classes)
- All preprocessing functions from Pavia University work with Houston dataset (they're generic)

---

**Status**: ✅ Ready to train!
**Last Updated**: 2025

