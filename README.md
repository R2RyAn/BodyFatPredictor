# 📉 Body Fat % Predictor

This project trains a deep learning model to predict human body fat percentage based on synthetic images.  
The dataset includes images covering a realistic range of body fat levels, generated and labeled automatically.

---

## 📂 **Project Overview**

**What was done:**
- ✅ Generated hundreds of images using **Stable Diffusion** with AUTOMATIC1111.
- ✅ Covered body fat levels from 10% to 50%.
- ✅ Used clear, explicit prompts to control physique realism and avoid over-muscular bias.
- ✅ Labeled each image in `metadata.csv` with `filename` → `body_fat_percent`.
- ✅ Wrote a custom PyTorch `Dataset` with robust missing file handling.
- ✅ Trained a simple regression model to predict body fat %.
- ✅ Saved model weights to `bodyfat_model.pth`.

---

## ⚙️ **How to Run**

```bash
# 1. Clone this repo
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
pip install -r requirements.txt

# 3. Mount dataset (if on Colab)
from google.colab import drive
drive.mount('/content/drive')

# 4. Train
python train.py

# 5. Model is saved to:
# bodyfat_model.pth
```
## 📈 Training Visualizations

### ✅ Training Loss Over Time

![Training Loss](https://github.com/R2RyAn/BodyFatPredictor/blob/master/test.png)

### ✅ True vs Predicted Body Fat %

![True vs Predicted](https://github.com/R2RyAn/BodyFatPredictor/blob/master/test.png)

---

## 🔑 Robust Features

- Handles missing images safely (`FileNotFoundError`)
- Skips bad CSV rows
- Logs average loss per epoch
- Plots loss curve at the end
- Saves final weights for reuse

---

## 🧩 How to Use Saved Model

```python
import torch

model = MyModelClass()  # replace with your actual model class
model.load_state_dict(torch.load("bodyfat_model.pth"))
model.eval()
