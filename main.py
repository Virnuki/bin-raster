import cv2
import numpy as np
import random as rd
from math import ceil


def create_empty(n: int, k: float) -> np.ndarray:
    return np.zeros((int(n * k), int(n * k)), dtype=np.uint8)


def add_bright(img: np.ndarray) -> np.ndarray:
    return np.array([list(map(lambda x: 255 * x, elem)) for elem in img])


def find_empties(n: int, k: float) -> set:
    length = int(n * k)
    diff = length - (n * int(k))
    int_part = int(k)
    step = ceil(n / (diff + 1))
    print((n / (diff + 1)), step)
    adds = n % (diff + 1)
    i, out = 0, set()
    while diff > 0:
        i += step * int_part
        if adds > 0:
            i += 1
            adds -= 1
        out |= {i}
        diff -= 1
    return out


def int_to_cluster(n: int, k: float) -> tuple:
    int_part = int(k)
    size = int_part * int_part
    count = min(n // (256 // (size + 1)), size)
    return tuple(map(int, ('1 ' * count + '0 ' * (size - count)).rstrip().split()))


def get_cluster(values: tuple, size: int) -> list:
    cluster = []
    rd.shuffle(list(values))
    for i in range(size):
        cluster.append(values[i * size:(i + 1) * size])
    return cluster


def fill_clusters(img: np.ndarray, new_img: np.ndarray, empties: set, n: int, k: float,
                  pattern=get_cluster) -> None:
    int_part = int(k)
    size = int(n * k)
    cl_ind = sorted((set(range(size)) - empties))[::int_part]
    for i in range(n):
        for j in range(n):
            values = pattern(int_to_cluster(img[i, j], k), int_part)
            new_img[cl_ind[i]:cl_ind[i] + int_part, cl_ind[j]:cl_ind[j] + int_part] = values


def fill_new_e_random(new_img: np.ndarray, empties: set) -> None:
    for i in range(len(new_img)):
        for j in range(len(new_img)):
            if i in empties or j in empties:
                new_img[i, j] = rd.randint(0, 1)


def interpolate(new_img: np.ndarray, empties: set, size: int) -> None:
    """
    НЕРАБОЧИЙ НЕЙРОСЛОП
    """
    for empty_row in sorted(empties):
        # Находим ближайшие непустые строки сверху и снизу
        top_row = empty_row - 1
        while top_row >= 0 and top_row in empties:
            top_row -= 1

        bottom_row = empty_row + 1
        while bottom_row < size and bottom_row in empties:
            bottom_row += 1
        # Интерполяция между найденными строками
        if bottom_row < size - 1:
            # Линейная интерполяция
            alpha = (empty_row - top_row) / (bottom_row - top_row)
            new_img[empty_row] = (1 - alpha) * new_img[top_row] + alpha * new_img[bottom_row]
        else:
            # Копируем нижнюю строку
            new_img[empty_row] = new_img[top_row]


def fill_new_i_interp(new_img: np.ndarray, empties: set) -> None:
    """
    Заполняет пустые строки и столбцы с помощью интерполяции соседних значений
    НАДО ПЕРЕПИСАТЬ ФУНКЦИЮ ИНТЕРПОЛЯЦИИ
    """
    if not empties:
        return
    size = new_img.shape[0]
    interpolate(new_img, empties, size)
    img_transposed = new_img.T
    interpolate(img_transposed, empties, size)
    new_img[:, :] = img_transposed.T


if __name__ == "__main__":
    # Baboo_256.tiff  Pepper_256.tiff
    img = cv2.imread(f'images/{input()}', cv2.IMREAD_GRAYSCALE)
    n, k = img.shape[0], float(input())
    new_image = create_empty(n, k)
    empt = find_empties(n, k)
    fill_clusters(img, new_image, empt, n, k)
    # fill_new_e_random(new_image, empt)
    fill_new_i_interp(new_image, empt)
    np.savetxt('output.csv', new_image, delimiter=',', fmt='%d')
    new_image = add_bright(new_image)
    cv2.imshow('Old image', img)
    cv2.imshow('New image', new_image)
    cv2.imwrite('output.png', new_image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
