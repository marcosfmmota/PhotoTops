#color_models.py

import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb
import math


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
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            # epsilon to avoid a division by zero
            ep = 0.001
            intensity = (r + g + b) / 3.0
            saturation = 1 - ((3 / (r + g + b)) * np.min(image[i, j, :]))
            rmg = r - g
            rmb = r - b
            gmb = g - b
            w = ((rmg + rmb) * 0.5) / (math.sqrt((rmg**2) + (rmb*gmb)) + ep)

            theta = math.acos(w)

            if b > g:
                theta = (2 * math.pi) - theta

            hsi = np.array([theta, saturation, intensity])

            # print(hsi)

            hsi_image[i, j, :] = hsi

    return hsi_image


def hsi_to_rgb(image):

    image = img_as_float(image)

    rgb_image = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            h = image[i, j, 0] * 360
            s = image[i, j, 1]
            i =image[i, j, 2]
            if 0 <= h < 120:

                b = i*(1 - s)
                r = i*(1 + ((s * math.cos(h)) / math.cos(60 - h)))
                g = 3*i - (r + b)
                rgb = np.array([r, g, b])
                rgb_image[i, j, :] = rgb

            elif 120 <= h < 240:
                h -= 120
                r = i * (1 - s)
                g = i * (1 + ((s * math.cos(h)) / math.cos(60 - h)))
                b = 3 * i - (r + g)
                rgb = np.array([r, g, b])
                rgb_image[i, j, :] = rgb

            else:
                h -= 240
                g = i * (1 - s)
                b = i * (1 + ((s * math.cos(h)) / math.cos(60 - h)))
                r = 3 * i - (g + b)
                rgb = np.array([r, g, b])
                rgb_image[i, j, :] = rgb

    rgb_image = img_as_ubyte(rgb_image)
    return rgb_image
