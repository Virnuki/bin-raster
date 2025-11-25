import cv2
import numpy as np
import random as rd
from math import ceil


def create_empty(n: int, k: float) -> np.ndarray:
    return np.zeros((int(n * k), int(n * k)), dtype="")


def add_bright(img: np.ndarray) -> np.ndarray:
    return np.array([list(map(lambda x: 255 * x, elem)) for elem in img])


def find_empties(n: int, k: float) -> set:
    length = int(n * k)
    diff = length - (n * int(k))
    int_part = int(k)
    step = int(n / (diff + 1))
    adds = ((n / (diff + 1)) - step) * (diff + 1)
    i, out = 0, set()
    while diff > 0:
        i += step * int_part
        if adds > 0:
            i += int_part
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
    print(cl_ind)
    for i in range(n):
        for j in range(n):
            values = pattern(int_to_cluster(img[i, j], k), int_part)
            new_img[cl_ind[i]:cl_ind[i] + int_part, cl_ind[j]:cl_ind[j] + int_part] = values


def fill_new_e_random(new_img: np.ndarray, empties: set) -> None:
    for i in range(len(new_img) - 1):
        for j in range(len(new_img) - 1):
            if i in empties or j in empties:
                new_img[i, j] = rd.randint(0, 1)


def sum_of_cluster(y0: int, x0: int, i_p: int, new_img: np.ndarray) -> int:
    count = 0
    for elem in new_img[y0:y0+i_p, x0:x0+i_p]:
        for num_elem in elem:
            count += num_elem
    return count
        

def fill_new_average(new_img: np.ndarray, empties: set, i_p: int) -> None:
    '''for rows'''
    for i in empties:
        for j in range(0, len(new_img), i_p):
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
                


if __name__ == "__main__":
    # Baboo_256.tiff  Pepper_256.tiff
    img = cv2.imread("images/Baboo_256.tiff", cv2.IMREAD_GRAYSCALE)
    n, k = img.shape[0], float(input())
    new_image = create_empty(n, k)
    empt = find_empties(n, k)
    z = list(empt)
    z.sort()
    print(z, len(new_image))
    fill_clusters(img, new_image, empt, n, k)
    #fill_new_e_random(new_image, empt)
    #fill_new_average(new_image, empt, int(k))
    np.savetxt('output.csv', new_image, delimiter=',', fmt='%d')
    new_image = add_bright(new_image)
    cv2.imwrite('output.png', new_image)
    cv2.imshow('Old image', img)
    cv2.imshow('New image', new_image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
