from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0

def build_transfer_model(num_classes, img_size=(224, 224)):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3)
    )
    # Fine-tune top layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    return model
