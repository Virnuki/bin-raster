import cv2
import numpy as np
import random as rd
from math import ceil


def create_empty(n: int, k: float) -> np.ndarray:
    return np.zeros((int(n * k), int(n * k)), dtype=np.uint8)


def add_bright(img: np.ndarray) -> np.ndarray:
    # Исправлено: преобразование в uint8 и правильное масштабирование
    return (img * 255).astype(np.uint8)


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
    # Исправлено: предотвращение деления на 0 и переполнения
    max_val = 256 // (size + 1) if size > 0 else 1
    count = min(n // max_val, size) if max_val > 0 else 0
    return tuple(map(int, ('1 ' * count + '0 ' * (size - count)).rstrip().split()))


def get_cluster(values: tuple, size: int) -> list:
    values_list = list(values)
    rd.shuffle(values_list)
    cluster = []
    for i in range(size):
        cluster.append(values_list[i * size:(i + 1) * size])
    return cluster


def fill_clusters(img: np.ndarray, new_img: np.ndarray, empties: set, n: int, k: float,
                  pattern=get_cluster) -> None:
    int_part = int(k)
    size = int(n * k)
    cl_ind = sorted((set(range(size)) - empties))[::int_part]
    for i in range(n):
        for j in range(n):
            values = pattern(int_to_cluster(img[i, j], k), int_part)
            # Исправлено: преобразование в numpy array с правильным типом
            cluster_data = np.array(values, dtype=np.uint8)
            new_img[cl_ind[i]:cl_ind[i] + int_part, cl_ind[j]:cl_ind[j] + int_part] = cluster_data


def fill_new_e_random(new_img: np.ndarray, empties: set) -> None:
    for i in range(len(new_img)):
        for j in range(len(new_img)):
            if i in empties or j in empties:
                new_img[i, j] = rd.randint(0, 1)


def fill_new_average(img: np.ndarray, new_img: np.ndarray, empties: set, k: float) -> None:
    # Исправлено: предотвращение переполнения и правильные границы
    img_h, img_w = img.shape
    new_h, new_w = new_img.shape
    
    for i in range(new_h):
        for j in range(new_w):
            if i in empties or j in empties:
                # Исправлено: правильное вычисление индексов и предотвращение выхода за границы
                ni = min(int(i / k), img_h - 1)
                nj = min(int(j / k), img_w - 1)
                
                if i in empties and j in empties:
                    # Исправлено: безопасное вычисление соседних пикселей
                    ni1 = min(ni + 1, img_h - 1)
                    nj1 = min(nj + 1, img_w - 1)
                    # Исправлено: правильное деление без переполнения
                    avg = (int(img[ni, nj]) + int(img[ni1, nj]) + int(img[ni, nj1]) + int(img[ni1, nj1])) // 4
                    new_img[i, j] = min(avg, 255)
                elif i in empties:
                    ni1 = min(ni + 1, img_h - 1)
                    avg = (int(img[ni, nj]) + int(img[ni1, nj])) // 2
                    new_img[i, j] = min(avg, 255)
                elif j in empties:
                    nj1 = min(nj + 1, img_w - 1)
                    avg = (int(img[ni, nj]) + int(img[ni, nj1])) // 2
                    new_img[i, j] = min(avg, 255)


if __name__ == "__main__":
    # Baboo_256.tiff  Pepper_256.tiff
    img = cv2.imread(f'images/{input()}', cv2.IMREAD_GRAYSCALE)
    n, k = img.shape[0], float(input())
    new_image = create_empty(n, k)
    empt = find_empties(n, k)
    fill_clusters(img, new_image, empt, n, k)
    # fill_new_e_random(new_image, empt)
    fill_new_average(img, new_image, empt, k)
    np.savetxt('output.csv', new_image, delimiter=',', fmt='%d')
    new_image = add_bright(new_image)
    cv2.imshow('Old image', img)
    cv2.imshow('New image', new_image)
    cv2.imwrite('output.png', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Исправлено: правильное имя функции
