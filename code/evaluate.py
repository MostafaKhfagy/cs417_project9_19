import os
from tensorflow.keras.models import load_model
from dataset import val_generator
from utils import plot_confusion_matrix, plot_sample_predictions
import numpy as np

MODEL_PATH = r"E:\Code\Fish Species Recognition\best_model_v10.h5"
RESULTS_DIR = r"E:\Code\Fish Species Recognition\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

model = load_model(MODEL_PATH)
print(f"[INFO] Loaded model from {MODEL_PATH}")

loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

class_names = list(val_generator.class_indices.keys())
y_true = val_generator.classes
y_pred = np.argmax(model.predict(val_generator, verbose=1), axis=1)

plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))

plot_sample_predictions(model, val_generator, class_names, save_path=os.path.join(RESULTS_DIR, "sample_predictions.png"), num_samples=10)



