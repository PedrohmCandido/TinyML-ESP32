import cv2
import numpy as np

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expanded_arr = np.expand_dims(gray_img, -1)
    return expanded_arr