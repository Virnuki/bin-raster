import cv2
import numpy as np


def create_empty(n: int, k: float) -> np.ndarray:
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


def int_to_cluster(n: int, k: float) -> tuple:
    int_part = int(k)
    size = int_part * int_part
    count = min(n // (256 // (size + 1)), size)
    print(count, size)
    return tuple(map(int, ('1 ' * count + '0 ' * (size - count)).rstrip().split()))


if __name__ == "__main__":
    # Baboo_256.tiff  Pepper_256.tiff
    img = cv2.imread(f'images/{input()}', cv2.IMREAD_GRAYSCALE)
    n, k = len(img), float(input())
    print(img)
