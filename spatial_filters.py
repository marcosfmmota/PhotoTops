from skimage import img_as_float
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math

def negative(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 255 - image[i, j]


def log_transform(image, c = 1.0):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = c * math.log2(1 + image[i, j])



def power_transform(image, c = 1.0, y = 1.0):

    image = img_as_float(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = c * math.pow(image[i, j], y)
    image = img_as_ubyte(image)
    return image


def bit_plane_slicing(image, bit_array):
    planes = 0
    for x in range(8):
        planes += bit_array[x]*(2**(8 - x))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = image[i, j] & planes


def linear_funtion(p1, p2):
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b =  p1[1] - (slope*p1[0])

    def lin_func(v):
        return slope * v + b

    return lin_func


def contrast_stretching(image, point1, point2):

    l_func1 = linear_funtion((0, 0), point1)
    l_func2 = linear_funtion(point1, point2)
    l_func3 = linear_funtion(point2, (255,255))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < point1[0]:
                image[i,j] = l_func1(image[i, j])
            elif image[i, j] > point2[0]:
                image[i, j] = l_func3(image[i, j])
            else:
                image[i, j] = l_func2(image[i, j])


def histogram(image):

    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1

    return hist


def show_histogram(hist):

    x_pos = np.arange(len(hist))
    plt.bar(x_pos, hist, width=0.9)
    plt.show()

def histogram_equalization(image):

    trans_func = histogram(image)
    acum = 0
    mn = image.size

    for x in range(len(trans_func)):
        acum += trans_func[x]
        trans_func[x] = (255 / mn) * acum

    trans_func = np.round(trans_func)


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = trans_func[image[i,j]]

    return image
