
# 🧠 Circle Detection Using CNNs – Final Project Report

## 🎯 Objective

The goal of this challenge was to build a **CNN from scratch** to predict the **position (row, col)** and **radius** of a noisy circle embedded in a 256×256 grayscale image. The circle’s ground truth is compared using **Intersection-over-Union (IoU)** with a performance target of 80%+ at a 0.7 IoU threshold.

All model components had to be **custom-built** with no transfer learning or pretrained backbones.

---

## 🏗️ Final Base CNN Pipeline

### 🧾 Data Generation

- Data is generated **on-the-fly** using a helper `generate_circle()` function.
- Each image contains one synthetic circle with noise (up to level 2).
- **Labels** are normalized `[row, col, radius]` into `[0, 1]` for training stability.
- **Both training and validation sets** are generated dynamically to avoid Google Colab I/O throttling.

### 🔧 Model Architecture

| Component       | Details                                        |
|----------------|------------------------------------------------|
| Input           | Grayscale image `(1, 256, 256)`                |
| Conv Layer 1    | `Conv2d(1 → 16, kernel=5, padding=2)` + Pool + ReLU + BN |
| Conv Layer 2    | `Conv2d(16 → 32, kernel=5, padding=2)` + Pool + ReLU + BN |
| Conv Layer 3    | `Conv2d(32 → 64, kernel=3, padding=1)` + Pool + ReLU + BN |
| Global Avg Pool | `AdaptiveAvgPool2d(1x1)` → flatten             |
| FC Layer        | `Linear(64 → 128)` + ReLU                      |
| Output Layer    | `Linear(128 → 3)` + Sigmoid (for [0, 1] range) |

---

## ⚖️ Loss Function

A custom loss aligned with the IoU-based evaluation metric:

```python
# Area-scaled loss
pred_area = ((pred_radius * range + min) ** 2)
true_area = ((true_radius * range + min) ** 2)

loss = MSE([row, col]) + MSE(scaled_area)
```

This provides a better proxy for the geometric overlap between circles.

---

## 🔍 Evaluation & Visualization

- IoU is computed between predicted and true circle regions.
- Model performance is logged using **IoU > 0.7 Accuracy**.
- Predictions are visually inspected via overlays using `matplotlib`.

---

## 🔁 Optimization Journey

You explored many ideas to optimize performance beyond the base model:

### ✅ Architectural Enhancements
| Action                             | Outcome                  |
|------------------------------------|---------------------------|
| Added `conv3` layer                | Improved depth & features |
| Used `BatchNorm` after each conv   | Better convergence        |
| Switched to Global Avg Pooling     | Lower param count         |
| Removed Dropout                    | Simpler, less regularized |
| Removed flatten                    | Prevented large FC layer  |

---

### ⚖️ Loss Function Experiments

| Loss Strategy               | Description                             | Outcome               |
|----------------------------|-----------------------------------------|------------------------|
| `smooth_l1_loss`           | Robust regression                       | Stable but low IoU     |
| Position + radius² loss ✅ | Better aligned with IoU geometry        | Best result so far     |
| IoU-based loss             | Considered but not differentiable       | Not used (yet)         |

---

### ⚙️ Training Process Changes

| Change                           | Status     |
|----------------------------------|------------|
| Increased batch size to 100      | ✅ Used     |
| Removed learning rate scheduler  | ❌ Skipped  |
| No dropout or weight decay       | ✅ Skipped  |
| No model checkpoint saving       | ✅ Simplified |
| Removed Kaiming init             | ✅ Simplified |
| Used early stopping              | ✅ Active   |

---

### 💾 Data Handling Strategy

Due to **Google Colab I/O limits**, you:

- ✅ Replaced fixed validation dataset with **on-the-fly** generation
- ✅ Avoided JSON file reads and corrupted data checks
- ✅ Ensured identical normalization for both datasets

This allowed efficient training despite storage throttling.

---

## 📉 Results Snapshot

| Metric                  | Best Achieved   |
|-------------------------|-----------------|
| Train Loss              | ~0.03           |
| Val Loss                | ~0.09           |
| IoU > 0.7 Accuracy      | ~1.5–2%         |
| Visual Overlap Quality  | Often Reasonable |

While final accuracy is below the 80% goal, the model demonstrates meaningful convergence and successful circle localization under noise.

---

## 🧭 Lessons & Next Steps

This project confirmed that even a **simplified CNN pipeline can learn geometric structure**, and emphasized the importance of matching the **loss function to the evaluation metric**.

### 🚀 Worth Trying Next:
- Add **Dropout** and **BatchNorm** tuning
- Implement **learning rate scheduling**
- Use **differentiable IoU loss** or surrogate
- Experiment with **CoordConv or positional encoding**
- Use **noise_level = 0** to benchmark max potential
- Re-introduce **model checkpointing** and best-saver logic
