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


def haar2d(image, levels=1):

    rows, cols = image.shape

    for k in range(levels):
        lev = 2 ** k

        lev_rows = rows // lev
        lev_cols = cols // lev
        for i in range(lev_rows):
            image[i, :] = haar1d(image[i, :])

        for j in range(lev_cols):
            image[:, j] = haar1d(image[:, j])

    return image


def haar2d_inverse(image, levels=1):

    rows, cols = image.shape

    for k in range(levels, 0, -1):

        for j in range(cols):
            image[:, j] = haar1d_inverse(image[:, j])

        for i in range(rows):
            image[i, :] = haar1d_inverse(image[i, :])

    return image


def haar_image(image, level=1):

    t_image = np.empty_like(image)

    r_comp = haar2d(image[:, :, 0], level)
    g_comp = haar2d(image[:, :, 1], level)
    b_comp = haar2d(image[:, :, 2], level)

    t_image[:, :, 0] = r_comp
    t_image[:, :, 1] = g_comp
    t_image[:, :, 2] = b_comp

    return t_image


def haar_inverse_image(image, level=1):

    t_image = np.empty_like(image)

    r_comp = haar2d_inverse(image[:, :, 0], level)
    g_comp = haar2d_inverse(image[:, :, 1], level)
    b_comp = haar2d_inverse(image[:, :, 2], level)

    t_image[:, :, 0] = r_comp
    t_image[:, :, 1] = g_comp
    t_image[:, :, 2] = b_comp

    return t_image
