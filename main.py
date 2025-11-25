import cv2
import numpy as np
import random as rd
from math import ceil


def create_empty(fin_size: int) -> np.ndarray:
    return np.zeros((fin_size, fin_size), dtype=np.uint8)


def add_bright(img: np.ndarray) -> np.ndarray:
    return np.array([list(map(lambda x: 255 * x, elem)) for elem in img])


def find_empties(n: int, k: float) -> set:
    length = int(n * k)
    diff = length - (n * int(k))
    int_part = int(k)
    step = int(n / (diff + 1))
    adds = n % (diff + 1)
    i, out = -1, set()
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


def average_fill(new_img: np.ndarray, empties: set, i_p: int) -> None:
    for i in empties:
        j = 0
        while j < len(new_img):
            if j in empties:
                j += 1
            else:
                sum_1 = sum_of_cluster(i - i_p, j, i_p, new_img)
                sum_2 = sum_of_cluster(i + 1, j, i_p, new_img)
                n = sum_1 + sum_2
                count = min(n // (i_p * i_p * 2 // (i_p + 1)), i_p)
                cort = list(map(int, ('1 ' * count + '0 ' * (i_p - count)).rstrip().split()))
                rd.shuffle(cort)
                new_img[i, j:j + i_p] = cort
            j += i_p


def fill_new_i_average(new_img: np.ndarray, empties: set, i_p: int) -> None:
    average_fill(new_img, empties, i_p)
    new_img = new_img.T
    average_fill(new_img, empties, i_p)


def sum_of_cluster(y0: int, x0: int, i_p: int, new_img: np.ndarray) -> int:
    count = 0
    for elem in new_img[y0:y0+i_p, x0:x0+i_p]:
        for num_elem in elem:
            count += num_elem
    return count


if __name__ == "__main__":
    # Baboo_256.tiff  Pepper_256.tiff
    # img = cv2.imread(f'images/{input()}', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(f'images/Baboo_256.tiff', cv2.IMREAD_GRAYSCALE)
    fin_size = int(input())
    n = img.shape[0]
    k = fin_size / n
    print(k)
    # n, k = img.shape[0], float(input())
    new_image = create_empty(fin_size)
    empt = find_empties(n, k)
    print(sorted(empt))
    fill_clusters(img, new_image, empt, n, k)
    #fill_new_e_random(img, empt)
    fill_new_i_average(new_image, empt, int(k))
    np.savetxt('output.csv', new_image, delimiter=',', fmt='%d')
    new_image = add_bright(new_image)
    cv2.imshow('Old image', img)
    cv2.imshow('New image', new_image)
    cv2.imwrite('output.png', new_image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
