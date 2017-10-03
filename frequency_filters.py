import math
import numpy as np
import spatial_filters as sf


def shrink_image(image, rate):

    new_shape = ((image.shape[0] // rate) + 1, (image.shape[1] // rate) + 1)
    shrinked = np.zeros(new_shape, dtype=np.uint8)
    print(image.shape)
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
