import numpy as np
from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
from skimage.exposure import equalize_hist

import spatial_filters as sf


def brightness_filter(image, percent):

    k = 1 - percent
    image_hsv = rgb2hsv(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            image_hsv[i, j, 2] *= k

    return hsv2rgb(image_hsv)


def average_filter_rgb(image, shape=(5, 5)):

    kernel = np.ones(shape)
    avg_image = np.empty_like(image)
    r_component = sf.convolve_average(image[:, :, 0], kernel)
    g_component = sf.convolve_average(image[:, :, 1], kernel)
    b_component = sf.convolve_average(image[:, :, 2], kernel)

    avg_image[:, :, 0] = r_component
    avg_image[:, :, 1] = g_component
    avg_image[:, :, 2] = b_component

    return avg_image


def average_filter_hsi(image, shape=(5, 5)):

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


def tone_filter_hsi(image, func):

    vfunc = np.vectorize(func)
    image[:, :, 2] = vfunc(image[:, :, 2])

    return image


def sepia_filter(image, c1=1.7, c2=1.3):

    image[:, :, 2] = image.sum(axis=2) / 3
    image[:, :, 0] = c1 * image[:, :, 2]
    mask = image[:, :, 0] > 1
    image[:, :, 0][mask] = 1
    image[:, :, 1] = c2 * image[:, :, 2]
    mask = image[:, :, 1] > 1
    image[:, :, 1][mask] = 1
    return image


def image_equalization(image):
    image = rgb2hsv(image)
    image[:, :, 2] = equalize_hist(image[:, :, 2])
    image = tone_filter_hsi(image, lambda x: x**0.93)
    image = hsv2rgb(image)
    return image


def chroma_key(image1, image2, radius=0.7, alpha=0.0):
    green = np.ones(image1.shape)
    green[:, :, :] = np.array([0.0, 1.0, 0.0])
    mask = ((image1 - green)**2).sum(axis=2) < radius ** 2
    for i in range(3):
        image1[:, :, i][mask] = image2[:, :, i][mask]
        image1[:, :, i][np.logical_not(mask)] += image2[:, :, i][np.logical_not(mask)]*alpha

    return image1

