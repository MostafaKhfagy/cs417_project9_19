import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dataset import train_generator, val_generator
from utils import plot_accuracy_curve, plot_loss_curve

RESULTS_DIR = r"E:\Code\Fish Species Recognition\results"

num_classes = len(train_generator.class_indices)

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(r"E:\Code\Fish Species Recognition\saved_model\best_model.h5", monitor='val_loss', save_best_only=True)
lr_reduce  = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=1e-7)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint, lr_reduce]
)

plot_accuracy_curve(history, save_path=os.path.join(RESULTS_DIR, "accuracy_curve.png"))
plot_loss_curve(history, save_path=os.path.join(RESULTS_DIR, "loss_curve.png"))
