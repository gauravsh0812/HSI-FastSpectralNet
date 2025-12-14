# 🛰️ HSI-FastSpectralNet
.

---

## 🧩 Project Structure

```
auto_modular_project/
├── data_loader.py      # Handles dataset loading and preprocessing
├── model.py            # Model definitions (CNN / Transformer / classifier)
├── train.py            # Training and evaluation routines
├── utils.py            # Helper utilities (metrics, visualization, etc.)
├── main.py             # Entry point — runs the entire pipeline
├── requirements.txt    # List of dependencies
```

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/auto_modular_project.git
cd auto_modular_project
pip install -r requirements.txt
```

> ✅ **Requires:** Python 3.8 or later

---

## 🚀 How to Run

Run the full workflow:

```bash
python main.py
```

This script will:
1. Download and load the **Pavia University** hyperspectral dataset  
2. Perform data preprocessing and summarization  
3. Optionally train a model (if training code is defined)  
4. Visualize key results (spectral bands, class maps, etc.)

---

## 🧠 Customize

You can extend or modify the pipeline easily:
- **`data_loader.py`** → change dataset source or preprocessing  
- **`model.py`** → define your own architecture (CNN, ViT, etc.)  
- **`train.py`** → modify training logic, optimizer, or evaluation metrics  
- **`utils.py`** → add custom metrics or helper functions  

Example:
```python
from data_loader import load_pavia_data
from model import MyCustomModel
```

---

## 📊 Dataset

Dataset used: [Pavia University Hyperspectral Image](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

---

## 🧾 License

MIT License © 2025 **Jayant Biradar**

---

## 🌟 Acknowledgements

Developed by **Jayant Biradar**  
Converted and modularized using AI-assisted notebook parsing.
