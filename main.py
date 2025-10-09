import cv2
import numpy as np


def create_empty(n, k):
    return np.zeros((int(n * k), int(n * k)), dtype=np.uint8)


if __name__ == "__main__":
    img = cv2.imread(f'images/{input()}', cv2.IMREAD_GRAYSCALE)
    n, k = len(img), float(input())
    print(create_empty(n, k))