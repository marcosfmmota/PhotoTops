from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
import spatial_filters as sf
import numpy as np


def subtract_two_images(image1, image2):

    sub_image = np.empty_like(image1)

    try:
        if image1.size != image2.size:
            raise NameError()

        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):

                sub_pixel = (image1[i, j, :] - image2[i, j, :])
                if sub_pixel.any() < 0:
                    sub_image[i, j, :] = np.zeros(3)
                else:
                    sub_image[i, j, :] = sub_pixel

        return sub_image

    except NameError:
        print("Images don't have the same size")


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
