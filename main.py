import cv2
import numpy as np


def create_empty(n: int, k: float) -> np.array:
    return np.zeros((int(n * k), int(n * k)), dtype=np.uint8)


def find_empties(n: int, k: float) -> set:
    length = int(n * k)
    diff = length - (n * int(k))
    step = int(k)
    count, i, out = 0, step, set()
    while diff > 0:
        out |= {i}
        if diff > 1:
            out |= {length - i - 1}
        i += step + 1
        diff -= 2
    return out


if __name__ == "__main__":
    # Baboo_256.tiff  Pepper_256.tiff
    img = cv2.imread(f'images/{input()}', cv2.IMREAD_GRAYSCALE)
    n, k = len(img), float(input())
    print(create_empty(n, k))
