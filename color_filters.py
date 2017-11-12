from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
import spatial_filters as sf
import numpy as np
from skimage.color import rgb2gray


def brightness_filter(image, percent):

    k = 1 - percent
    image_hsv = rgb2hsv(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            image_hsv[i, j, 2] *= k

    return hsv2rgb(image_hsv)


def average_filter_rgb(image, shape=(3, 3)):

    kernel = np.ones(shape)
    avg_image = np.empty_like(image)
    r_component = sf.convolve_average(image[:, :, 0], kernel)
    g_component = sf.convolve_average(image[:, :, 1], kernel)
    b_component = sf.convolve_average(image[:, :, 2], kernel)

    avg_image[:, :, 0] = r_component
    avg_image[:, :, 1] = g_component
    avg_image[:, :, 2] = b_component

    return avg_image


def average_filter_hsi(image, shape=(3,3)):

    kernel = np.ones(shape)
    avg_image = np.empty_like(image)
    i_component = sf.convolve_average(image[:, :, 2], kernel)

    avg_image[:, :, 0] = image[:, :, 0]
    avg_image[:, :, 1] = image[:, :, 1]
    avg_image[:, :, 2] = i_component

    return avg_image


def tone_filter_rgb(image, func):

    tone_image = np.empty_like(image)
    vfunc = np.vectorize(func)
    r_component = vfunc(image[:, :, 0])
    g_component = vfunc(image[:, :, 1])
    b_component = vfunc(image[:, :, 2])

    tone_image[:, :, 0] = r_component
    tone_image[:, :, 1] = g_component
    tone_image[:, :, 2] = b_component

    return tone_image


def sepia_filter(image, c1=1.8, c2=1.4):

    image[:, :, 2] = image.sum(axis=2) / 3
    image[:, :, 0] = c1 * image[:, :, 2]
    mask = image[:, :, 0] > 1
    image[:, :, 0][mask] = 1
    image[:, :, 1] = c2 * image[:, :, 2]
    mask = image[:, :, 1] > 1
    image[:, :, 1][mask] = 1
    return image
