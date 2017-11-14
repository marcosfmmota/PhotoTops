# color_models.py

import math

import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte


def rgb_to_grayscale(image):

    gray_image = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]

            gray = (r + g + b) / 3.0
            gray_image[i, j] = gray

    return gray_image


def rgb_to_cmy(image):

    image = img_as_float(image)

    cmk_image = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cmk = 1 - image[i, j, :]
            cmk_image[i, j, :] = cmk

    cmk_image = img_as_ubyte(cmk_image)

    return cmk_image


def rgb_to_hsi(image):

    image = img_as_float(image)

    hsi_image = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j, :]
            # epsilon to avoid a division by zero
            ep = 0.000001
            intensity = (r + g + b) / 3.0
            saturation = 1 - ((3 * min([r, g, b])) / (r + g + b))
            rmg = r - g
            rmb = r - b
            gmb = g - b
            w = ((rmg + rmb) * 0.5) / (math.sqrt((rmg ** 2) + (rmb * gmb)) + ep)
            hue = math.acos(w)/(2*math.pi)

            if b > g:
                hue = 1 - (hue/(2*math.pi))
            # print(theta)
            hsi = np.array([hue, saturation, intensity])

            # print(hsi)

            hsi_image[i, j, :] = hsi

    return hsi_image


def hsi_to_rgb(image):

    rgb_image = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            hue = image[i, j, 0] * (2*math.pi)
            sat = image[i, j, 1]
            intens = image[i, j, 2]

            if 0 <= hue < math.radians(120):

                b = intens*(1 - sat)
                r = intens*(1 + ((sat * math.cos(hue)) / math.cos(math.radians(60) - hue)))
                g = (3 * intens) - (r + b)
                rgb = np.array([r, g, b])
                rgb_image[i, j, :] = rgb

            elif math.radians(120) <= hue < math.radians(240):
                hue -= math.radians(120)
                r = intens * (1 - sat)
                g = intens * (1 + ((sat * math.cos(hue)) / math.cos(math.radians(60) - hue)))
                b = (3 * intens) - (r + g)
                rgb = np.array([r, g, b])
                rgb_image[i, j, :] = rgb

            else:
                hue -= math.radians(240)
                g = intens * (1 - sat)
                b = intens * (1 + ((sat * math.cos(hue)) / math.cos(math.radians(60) - hue)))
                r = (3 * intens) - (g + b)
                rgb = np.array([r, g, b])
                rgb_image[i, j, :] = rgb
                # print(rgb)

    return rgb_image
