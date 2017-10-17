# restoration_filters.py
import numpy as np
from skimage import util
import spatial_filters as sf


def arithmetic_mean_filter(image, size=(3, 3)):

    kernel = np.ones(size)
    sf.convolve_average(image, kernel)

    return image


def geometric_mean_filter(image, size=(3, 3)):

    kernel = np.ones(size)
    mn = size[0]*size[1]
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            conv_pixel = 1.0
            for a in range(-n_rows, n_rows + 1):
                for b in range(-n_cols, n_cols + 1):
                    aux_pix = padded_image[i + a, j + b]
                    conv_pixel *= aux_pix
            conv_pixel = conv_pixel ** (1/mn)
            image[i - n_rows, j - n_cols] = conv_pixel

    return image


def harmonic_mean_filter(image, size=(3, 3)):

    kernel = np.ones(size)
    mn = size[0]*size[1]
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            conv_pixel = 0.0
            for a in range(-n_rows, n_rows + 1):
                for b in range(-n_cols, n_cols + 1):
                    aux_pix = padded_image[i + a, j + b]
                    conv_pixel += 1 / aux_pix
            conv_pixel = mn / conv_pixel
            image[i - n_rows, j - n_cols] = conv_pixel

    return image


def contraharmonic_mean_filter(image, size=(3, 3), order=0):

    kernel = np.ones(size)
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            conv_pixel1 = 0.0
            conv_pixel2 = 0.0
            for a in range(-n_rows, n_rows + 1):
                for b in range(-n_cols, n_cols + 1):
                    aux_pix = padded_image[i + a, j + b]
                    conv_pixel1 += aux_pix ** (order + 1)
                    conv_pixel2 += aux_pix ** order
            # print(conv_pixel1)
            # print(conv_pixel2)

            conv_pixel = conv_pixel1 / conv_pixel2
            # print(conv_pixel)
            image[i - n_rows, j - n_cols] = conv_pixel

    return image


def median_filter(image, size=(3, 3)):

    kernel = np.ones(size)
    image = sf.convolve_median(image, kernel)

    return image


def max_filter(image, size=(3, 3)):

    kernel = np.ones(size)
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            conv_pixel = 0.0
            for a in range(-n_rows, n_rows + 1):
                for b in range(-n_cols, n_cols + 1):
                    aux_pix = padded_image[i + a, j + b]
                    if aux_pix > conv_pixel:
                        conv_pixel = aux_pix
            image[i - n_rows, j - n_cols] = conv_pixel

    return image


def min_filter(image, size=(3, 3)):

    kernel = np.ones(size)
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            conv_pixel = 255.0
            for a in range(-n_rows, n_rows + 1):
                for b in range(-n_cols, n_cols + 1):
                    aux_pix = padded_image[i + a, j + b]
                    if aux_pix < conv_pixel:
                        conv_pixel = aux_pix
            image[i - n_rows, j - n_cols] = conv_pixel

    return image


def alpha_trimmed_filter(image, d=0, size=(3, 3)):

    kernel = np.ones(size)
    mn = size[0]*size[1]
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            crop = padded_image[i - n_rows:i + n_rows + 1, j - n_cols: j + n_cols + 1]
            # print(crop)
            crop = np.reshape(crop, mn)
            crop = np.sort(crop)
            crop = crop[d // 2: -(d // 2)]
            conv_pixel = (1 / (mn - d)) * np.sum(crop)
            image[i - n_rows, j - n_cols] = conv_pixel

    return image


def mean_adaptive_filter(image, variance, size=(3, 3)):

    kernel = np.ones(size)
    mn = size[0]*size[1]
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            crop = padded_image[i - n_rows:i + n_rows + 1, j - n_cols: j + n_cols + 1]
            # print(crop)
            crop = np.reshape(crop, mn)
            variance_local = crop.var()
            mean_local = crop.mean()
            # print(variance_local)
            pixel = image[i - n_rows, j - n_cols]
            image[i - n_rows, j - n_cols] = pixel - ((variance/variance_local)*(pixel - mean_local))

    return image


def median_adaptive_filter(image, max_size, size=(3, 3)):

    max_n_rows = max_size[0] // 2
    max_n_cols = max_size[1] // 2
    padded_image = util.pad(image, ((max_n_rows, max_n_rows), (max_n_cols, max_n_cols)), 'constant', constant_values=0)

    for i in range(max_n_rows, image.shape[0] + max_n_rows):
        for j in range(max_n_cols, image.shape[1] + max_n_cols):
            # print((i, j))
            new_size = size
            while new_size <= max_size:
                n_rows = new_size[0] // 2
                n_cols = new_size[1] // 2
                mn = new_size[0] * new_size[1]
                # print("d")
                crop = padded_image[i - n_rows:i + n_rows + 1, j - n_cols: j + n_cols + 1]
                # print(crop)
                crop = np.reshape(crop, mn)
                Zmax = crop.max()
                Zmin = crop.min()
                Zmed = np.median(crop)
                Zxy = padded_image[i, j]

                A1 = Zmed - Zmin
                A2 = Zmed - Zmax

                if A1 > 0 and A2 < 0:
                    B1 = Zxy - Zmin
                    B2 = Zxy - Zmax
                    if B1 > 0 and B2 < 0:
                        break
                    else:
                        # image[i - n_rows, j - n_cols] = Zmed
                        padded_image[i, j] = Zmed
                        break
                else:
                    new_size = (new_size[0] + 2, new_size[1] + 2)
                    if new_size < max_size:
                        pass
                    else:
                        # image[i - n_rows, j - n_cols] = Zmed
                        padded_image[i, j] = Zmed
                        break

    return padded_image[max_n_rows:-max_n_rows, max_n_cols: - max_n_rows]
