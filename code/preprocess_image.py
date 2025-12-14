import cv2
import numpy as np
from rembg import remove
from PIL import Image

def preprocess_fish_image(image_path, save_path):

    input_image = Image.open(image_path).convert("RGBA")
    no_bg = remove(input_image)
    no_bg = np.array(no_bg)

    alpha = no_bg[:, :, 3]
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    cropped = no_bg[y:y+h, x:x+w]

    scale = 224 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((224, 224, 4), dtype=np.uint8)
    sx = (224 - new_w) // 2
    sy = (224 - new_h) // 2
    canvas[sy:sy+new_h, sx:sx+new_w] = resized

    Image.fromarray(canvas).save(save_path, format="PNG")

