# Project 9: Fish Species Recognition
## Team 1

## Project Overview
This project implements a Convolutional Neural Network (CNN) for fish species recognition from images.

## Team Members
- [Mostafa Khfagy] - ID: [2127109]
- [ahmed mousaad] - ID: [2127063]
- [ahmed saber hussien] - ID: [2127394]
- [mohamed ahmed] - ID: [1927061]
- [ziad mohamed] - ID: [1727193]

## Dataset
The dataset consists of 5 fish species with 5,000 total images.
- Salmon, Tuna, Bass, Trout, Cod
- 70% Training, 15% Validation, 15% Test

## Installation

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare your data
Place your fish images in the appropriate folders:
- `data/train/<species>/` for training
- `data/val/<species>/` for validation
- `data/test/<species>/` for testing

### 2. Train the model
```bash
cd code
python train.py
```

### 3. Evaluate the model
```bash
cd code
python evaluate.py
```

### 4. Make predictions
```bash
cd code
python predict.py --image path/to/your/image.jpg
```

## Model Architecture
- Input: 128x128 RGB images
- CNN layers: 3 Conv2D + MaxPooling2D
- Dense layers: 256 units with Dropout
- Output: 5 classes (softmax)

## Results
- Overall Accuracy: [To be filled]
- Best model saved in: `saved_model/best_model.h5`
- Results saved in: `results/`

## Files Structure
```
Project_9_1/
├── README.md
├── requirements.txt
├── code/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── dataset.py
│   ├── utils.py
│   └── predict.py
├── saved_model/
│   └── best_model.h5
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── data/
│   ├── train/
│   ├── val/
│   └── test/
└── documentation/
    └── dataset_info.md
```

