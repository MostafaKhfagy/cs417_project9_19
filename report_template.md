# Fish Species Recognition Using Convolutional Neural Networks


### Team Members
 [Mostafa Khfagy] - ID: [2127109]
- [ahmed mousaad] - ID: [2127063]
- [ahmed saber hussien] - ID: [2127394]
- [mohamed ahmed] - ID: [1927061]
- [ziad mohamed] - ID: [1727193]

### Submission Date: [14/12/2025]

---

## 1. Introduction

Automatic identification of fish species is an important task in fisheries management, conservation, and ecological studies. Manual identification is time-consuming, error-prone, and requires expert knowledge. Using deep learning for fish species recognition can accelerate classification, reduce human error, and support research in marine biology.

## 2. What is the Problem?

Identifying fish species manually is a challenging and time-consuming task. It requires expert knowledge, and even experienced researchers can make mistakes when distinguishing between visually similar species. This makes large-scale monitoring of fish populations inefficient and prone to errors.

## 3. Why is it Important?

Accurate identification of fish species is crucial for fisheries management, conservation efforts, and ecological research. Correct species classification helps track biodiversity, detect endangered species, and make informed decisions for sustainable fishing practices. Automating this process with deep learning can save time, reduce human error, and support scientific studies in marine biology.

## 4. Dataset

### 3.1 Dataset Source
- **Source:** https://data.mendeley.com/datasets/2gkj4h388d/3
- **Total species:** 12 fish species
- **Average images per class:** 300

### 3.2 Data Statistics
| Characteristic | Value |
|---|---|
| Number of Classes | 12 |
| Images per Class (average) | 300 |
| Input Resolution | 224×224 pixels |
| Total Approximate Images | 3,600 |

### 3.3 Data Preprocessing
1. **Resizing:** All images resized to 224×224 pixels
2. **Normalization:** Pixel values normalized to standard range
3. **Background Removal:** Applied to focus on fish features
4. **Train-Validation Split:** 95% training, 5% validation

## 5. Methodology

### 4.1 CNN Architecture
We used **EfficientNetB0** as the backbone feature extractor, followed by fully connected dense layers for classification into 12 fish species.

**Architecture Summary:**
```
Input: 224×224×3 (RGB image)
│
├── EfficientNetB0 (pretrained, include_top=False)
│
├── GlobalAveragePooling2D
│
├── Dense(512) → BatchNorm → ReLU → Dropout(0.4)
│
├── Dense(256) → BatchNorm → ReLU → Dropout(0.3)
│
├── Dense(128) → BatchNorm → ReLU → Dropout(0.2)
│
└── Dense(12, softmax)
```

### 4.2 Data Augmentation
Applied to training data to improve model generalization:
- Rotation: ±15 degrees
- Width/Height shift: ±10%
- Shear: 5%
- Zoom: ±15%
- Horizontal flip
- Brightness: 0.8–1.2
- Channel shift: ±10

### 4.3 Training Configuration
| Hyperparameter       | Value     |
|----------------------|-----------|
| Train-test-Validation  | 70%-15%-15%  |
| Optimizer            | Adam      |
| Learning Rate        | 1e-4      |
| Loss Function        | Categorical Cross-Entropy |
| Batch Size           | 32        |
| Epochs               | 20        |
| Early Stopping       | Applied   |
| Metrics              | Accuracy & F1-score |

### 4.4 Callbacks Used
1. **EarlyStopping:** Prevents overfitting
2. **ModelCheckpoint:** Saves best model based on validation accuracy
3. **Learning Rate Adjustment:** Dynamic reduction during training

## 6. Results

### 6.1 Validation Performance
- **Validation Loss:** 0.1254
- **Validation Accuracy:** 95.88%

### 6.2 Per-Class Performance (F1-Scores)
| Class | F1-Score |
|-------|----------|
| Class 1  | 0.94 |
| Class 2  | 0.92 |
| Class 3  | 0.95 |
| Class 4  | 0.93 |
| Class 5  | 0.91 |
| Class 6  | 0.94 |
| Class 7  | 0.90 |
| Class 8  | 0.96 |
| Class 9  | 0.92 |
| Class 10 | 0.93 |
| Class 11 | 0.91 |
| Class 12 | 0.95 |

### 5.3 Visualizations
1. **Confusion Matrix:** Reveals misclassification patterns between species (see confusion_matrix.png)
2. **Sample Predictions:** Visual examples of model predictions on test data (see sample_predictions.png)

## 6. Discussion

### 6.1 What Worked Well
1. **EfficientNetB0 Backbone:** Efficiently extracted features with pretrained weights
2. **Data Augmentation:** Improved model generalization and robustness
3. **Background Removal:** Helped the model focus on relevant fish features
4. **Transfer Learning:** Leveraging pretrained weights accelerated training and improved performance

### 6.2 Model Limitations
1. **Visual Similarity:** Some misclassifications occur between visually similar fish species
2. **Dataset Size:** Relatively small (~300 images per class) may limit generalization
3. **Generalization:** Model may not perform well on completely new or different datasets
4. **Environmental Variation:** Performance depends on consistent lighting and image quality

### 6.3 Future Improvements
1. **Increase Dataset Size:** Collect more diverse images, especially for rare species
2. **Experiment with Architectures:** Test EfficientNetB3, ResNet50, or Vision Transformers
3. **Advanced Techniques:** Apply segmentation and attention mechanisms
4. **Ensemble Methods:** Combine multiple models for improved robustness
5. **Field Deployment:** Optimize for real-time inference on mobile or edge devices

## 7. Conclusion and Future Work

### 7.1 Conclusion
The implemented model achieved **93.88% validation accuracy** with strong F1-scores across all 12 fish species (0.90–0.96). The use of EfficientNetB0 as a backbone with transfer learning proved highly effective for automated fish species classification. This demonstrates the viability of deep learning for automated species identification in fisheries management, conservation, and ecological research.

### 7.2 Future Work
1. **Dataset Expansion:** Increase dataset size and diversity for improved generalization
2. **Advanced Architectures:** Experiment with EfficientNetB3, ResNet50, and Vision Transformers
3. **Segmentation & Attention:** Apply object detection and attention mechanisms for improved feature focus
4. **Field Deployment:** Develop mobile or web applications for real-time fish identification in the field
5. **Ensemble Learning:** Combine multiple models for increased robustness and accuracy

## 8. References

1. Kaggle Fish Species Dataset. [Link]
2. TensorFlow Documentation. [Link]
3. Keras Documentation. [Link]
4. Research papers on marine species classification

**Note:** All code, models, and results are available in the submitted project folder.
