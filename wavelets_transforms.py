import numpy as np


def haar1d(array):
    temp = np.empty_like(array)
    # coefficients
    s0 = 0.5
    s1 = 0.5
    w0 = 0.5
    w1 = -0.5
    h = array.size // 2
    for i in range(h):
        k = i * 2
        temp[i] = (array[k] * s0) + (array[k + 1] * s1)
        temp[i + h] = (array[k] * w0) + (array[k + 1] * w1)

    return temp


def haar1d_inverse(array):
    temp = np.empty_like(array)
    # coefficients
    s0 = 0.5
    s1 = 0.5
    w0 = 0.5
    w1 = -0.5
    h = array.size // 2

    for i in range(h):
        k = i * 2
        temp[k] = ((array[i] * s0) + (array[i + h] * w0)) / w0
        temp[k + 1] = ((array[i] * s1) + (array[i + h] * w1)) / s1

    return temp


def haar2d(image):

    rows, cols = image.shape

    for i in range(rows):
            image[i, :] = haar1d(image[i, :])

    for j in range(cols):
            image[:, j] = haar1d(image[:, j])

    return image


def haar2d_inverse(image):

    rows, cols = image.shape

    for j in range(cols):
        image[:, j] = haar1d_inverse(image[:, j])

    for i in range(rows):
        image[i, :] = haar1d_inverse(image[i, :])

    return image


def haar_image(image, level=1):

    t_image = np.empty_like(image)

    for i in range(level):
        rows, cols, _ = image.shape
        rows //= 1 + i
        cols //= 1 + i

        r_comp = haar2d(image[:rows, :cols, 0])
        g_comp = haar2d(image[:rows, :cols, 1])
        b_comp = haar2d(image[:rows, :cols, 2])

        t_image[:rows, :cols, 0] = r_comp
        t_image[:rows, :cols, 1] = g_comp
        t_image[:rows, :cols, 2] = b_comp

    return t_image


def haar_inverse_image(image, level=1):

    t_image = np.empty_like(image)

    for i in range(level, 0, -1):
        rows, cols, _ = image.shape
        rows = rows // i
        cols = cols // i

        r_comp = haar2d_inverse(image[:rows, :cols, 0])
        g_comp = haar2d_inverse(image[:rows, :cols, 1])
        b_comp = haar2d_inverse(image[:rows, :cols, 2])

        t_image[:rows, :cols, 0] = r_comp
        t_image[:rows, :cols, 1] = g_comp
        t_image[:rows, :cols, 2] = b_comp

    return t_image
