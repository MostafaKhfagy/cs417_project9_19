import os
import random
from preprocess_image import preprocess_fish_image

RAW_DATASET = r"E:\Code\Fish Species Recognition\dataset\BD-Freshwater-Fish"
OUT_DATASET = r"E:\Code\Fish Species Recognition\dataset_preprocessed"

SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
random.seed(42)

for split in SPLITS:
    os.makedirs(os.path.join(OUT_DATASET, split), exist_ok=True)

for cls in os.listdir(RAW_DATASET):
    cls_path = os.path.join(RAW_DATASET, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    split_map = {
        "train": images[:n_train],
        "val":   images[n_train:n_train+n_val],
        "test":  images[n_train+n_val:]
    }

    for split, imgs in split_map.items():
        out_cls = os.path.join(OUT_DATASET, split, cls)
        os.makedirs(out_cls, exist_ok=True)

        for img in imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(out_cls, os.path.splitext(img)[0] + ".png")
            preprocess_fish_image(src, save_path=dst)
