# Slide 1 — Title
**CSC3218 Deep Learning Project-Based Exam**  
**CIFAR-10 Image Classification Using a CNN**

- Student: Obba Mark Calvin  
- Reg No: S23B23/047  
- Access No: B24277  
- Course: Deep Learning (BSCS 3:2)  
- Semester: Easter 2026

---

# Slide 2 — Problem Statement
**Problem:** Many computer vision systems fail when images are low-resolution, noisy, or visually ambiguous.  
In practical settings (e.g., surveillance, edge devices, robotics), we need a model that can still identify objects reliably from very small images.

- CIFAR-10 represents this challenge with `32 x 32` images and 10 similar object categories.
- Key difficulty: inter-class similarity (e.g., cat vs dog, deer vs horse, automobile vs truck).
- The problem to solve is not just classification, but **robust and generalizable** classification under limited visual detail.
- Therefore, we design and evaluate a CNN pipeline that minimizes overfitting while maintaining strong test performance.

---

# Slide 3 — Dataset Description
**CIFAR-10 Dataset**

- Total images: `60,000`
- Training images: `50,000`
- Test images: `10,000`
- Classes: 10 balanced categories
- Preprocessing:
  - Pixel normalization from `[0,255]` to `[0,1]`
  - Train/validation split from training set:
    - Train: `40,000`
    - Validation: `10,000`

---

# Slide 4 — Data Augmentation
**Why augmentation?** Improve generalization and reduce overfitting.

- `ZeroPadding2D(4)` + `RandomCrop(32,32)` (translation robustness)
- `RandomFlip(horizontal)` (orientation robustness)
- `RandomRotation(0.08)` (small viewpoint changes)

**Benefit:** Model sees varied but label-preserving samples, improving test-time performance.

---

# Slide 5 — Model Architecture
**CNN Design (Keras/TensorFlow)**

- Input `32 x 32 x 3` + augmentation block
- **Block 1:** Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout(0.2)
- **Block 2:** Conv(128) -> BN -> ReLU -> Conv(128) -> BN -> ReLU -> MaxPool -> Dropout(0.3)
- **Block 3:** Conv(256) -> BN -> ReLU -> Conv(256) -> BN -> ReLU
- GlobalAveragePooling -> Dropout(0.4) -> Dense(10, Softmax)

**Insert figure:** `cifar10_cnn_architecture.png`

---

# Slide 6 — Training Procedure
**Hyperparameters & Setup**

- Framework: TensorFlow / Keras
- Optimizer: AdamW (fallback Adam)
- Initial learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: `128`
- Max epochs: `80`
- Loss: Sparse Categorical Cross-Entropy

**Training strategies**
- Early stopping (`restore_best_weights=True`)
- ReduceLROnPlateau
- Dropout + augmentation + weight decay regularization

---

# Slide 7 — Results (Overall Metrics)
**Test Set Performance**

- **Accuracy:** `0.8858`
- **Precision (Macro):** `0.8860`
- **Recall (Macro):** `0.8858`
- **F1 Score (Macro):** `0.8854`

**Visual outputs to show**
- Accuracy/Loss learning curves
- Confusion matrix
- Sample predictions (one per class)

---

# Slide 8 — Performance Analysis
**Model behavior**

- Training and validation curves show stable learning
- Regularization strategies controlled overfitting
- Strong and balanced macro metrics indicate good class-level consistency

**Common confusion patterns**
- Similar visual classes are harder:
  - cat vs dog
  - deer vs horse
  - automobile vs truck

---

# Slide 9 — Improvements
**Possible next steps**

- Use stronger architectures (e.g., ResNet variants)
- Apply advanced augmentation (MixUp, Cutout, RandAugment)
- Perform systematic hyperparameter tuning
- Try test-time augmentation / model ensembling
- Analyze hard examples and class-specific errors

---

# Slide 10 — Conclusion
This project delivered an end-to-end CNN pipeline for CIFAR-10 classification using TensorFlow/Keras.  
The final model achieved strong results (**Acc 0.8858, Macro F1 0.8854**) with good generalization through augmentation, dropout, weight decay, LR scheduling, and early stopping.  
Overall, the model is accurate, interpretable, and a strong baseline for further improvement.

