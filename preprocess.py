import cv2
import numpy as np

def load_and_preprocess(image_path, size=8):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img
