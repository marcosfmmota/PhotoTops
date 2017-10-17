import math
import numpy as np
import spatial_filters as sf
from skimage.util import pad
from scipy.spatial import distance
import math

def shrink_image(image, rate):

    new_shape = ((image.shape[0] // rate) + 1, (image.shape[1] // rate) + 1)
    shrinked = np.zeros(new_shape, dtype=np.uint8)
    # print(image.shape)
    for i in range(0, image.shape[0], rate):
        for j in range(0, image.shape[1], rate):
            # print((i // rate, j // rate))
            shrinked[i // rate, j // rate] = image[i, j]

    return shrinked[:-2, :-2]


def shrink_average_image(image, rate):

    kernel = np.ones((3, 3))

    average_image = sf.convolve_average(image, kernel)
    shrinked = shrink_image(average_image, rate)

    return shrinked

def compute_spectrum(image):

    image = image.astype(np.float)

    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         image[i, j] = (-1 ** (i + j)) * image[i, j]
    #         print(image[i, j])

    spectrum = np.fft.fft2(image)
    spectrum = np.fft.fftshift(spectrum)
    spectrum = np.abs(spectrum)
    sf.log_transform(spectrum)

    return spectrum


def frequency_distance(spectrum):

    d_point = [spectrum.shape[0] / 2, spectrum.shape[1] / 2]

    dist = np.ones(spectrum.shape)

    for u in range(spectrum.shape[0]):
        for v in range(spectrum.shape[1]):
            dist[u, v] = math.sqrt((u - d_point[0])**2 + (v - d_point[1])**2)

    return dist


def general_frequency_filter(image, func):

    padded_image = pad(image, ((0, image.shape[0]), (0, image.shape[1])), 'constant', constant_values=0)
    # print(padded_image.shape)
    spectrum = np.fft.fft2(padded_image)
    spectrum = np.fft.fftshift(spectrum)
    # print(spectrum.shape)
    dist = frequency_distance(spectrum)
    H = dist.astype(np.float)

    vfunc = np.vectorize(func)
    H = vfunc(H)
    spectrum = np.multiply(H, spectrum)
    spectrum = np.fft.ifftshift(spectrum)
    spectrum = np.fft.ifft2(spectrum)
    spectrum = np.abs(spectrum)
    image_filtered = spectrum[:image.shape[0], :image.shape[1]]

    return image_filtered

