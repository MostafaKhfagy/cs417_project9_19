# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, figsize=(10, 10), cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=cmap, xticks_rotation='vertical', colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved at {save_path}")


def plot_sample_predictions(model, generator, class_names, save_path, num_samples=10, figsize=(15, 5)):
    x, y_true = next(generator)
    y_pred_probs = model.predict(x, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

    num_samples = min(num_samples, len(x))
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    
    for i in range(num_samples):
        axes[i].imshow(x[i].astype("uint8"))
        axes[i].axis('off')
        axes[i].set_title(f"T: {class_names[y_true_labels[i]]}\nP: {class_names[y_pred[i]]}", fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Sample predictions saved at {save_path}")

def plot_accuracy_curve(history, save_path):

    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Accuracy curve saved to {save_path}")


def plot_loss_curve(history, save_path):

    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Loss curve saved to {save_path}")
