import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset import train_generator, val_generator
from model import build_transfer_model
from utils import plot_accuracy_curve, plot_loss_curve


RESULTS_DIR = r"E:\Code\Fish Species Recognition\results"
MODEL_DIR = r"E:\Code\Fish Species Recognition\saved_model"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

num_classes = len(train_generator.class_indices)

model = build_transfer_model(num_classes)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.h5"),
    monitor='val_loss',
    save_best_only=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.2,
    min_lr=1e-7
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint, lr_reduce]
)

plot_accuracy_curve(
    history,
    save_path=os.path.join(RESULTS_DIR, "accuracy_curve.png")
)

plot_loss_curve(
    history,
    save_path=os.path.join(RESULTS_DIR, "loss_curve.png")
)
