import math

import numpy as np
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage import img_as_int
from skimage import exposure
from skimage import util

def negative(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 255 - image[i, j]


def log_transform(image, c=1.0):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = c * math.log2(1 + image[i, j])


def power_transform(image, c=1.0, y=1.0):

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

    return image


def linear_funtion(p1, p2):
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - (slope*p1[0])

    def lin_func(v):
        return slope * v + b

    return lin_func


def contrast_stretching(image, point1, point2):

    l_func1 = linear_funtion((0, 0), point1)
    l_func2 = linear_funtion(point1, point2)
    l_func3 = linear_funtion(point2, (255, 255))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < point1[0]:
                image[i, j] = l_func1(image[i, j])
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
            image[i, j] = trans_func[image[i, j]]

    return image


def local_equalization(image):

    shape = image.shape

    padded_image = np.zeros((shape[0]+2, shape[1]+2), dtype=np.uint8)
    # padded_image[-1:1,-1:1] = 255
    padded_image[1:shape[0]+1, 1:shape[1]+1] = image
    for i in range(1, shape[0]+1):
        for j in range(1, shape[1]+1):
            local_im = padded_image[i-1:i+2, j-1:j+2]
            padded_image[i-1:i+2, j-1:j+2] = histogram_equalization(local_im)
    equalized_image = padded_image[1:shape[0]+1, 1:shape[1]+1]
    return equalized_image


def convolve2d(image, kernel):

    kernel = np.fliplr(kernel)
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0]+n_rows):
        for j in range(n_cols, image.shape[1]+n_cols):

            # print(padded_image[i,j])
            # image_array = []
            conv_pixel = 0
            # print("#")
            for a in range(-1, 2):
                for b in range(-1, 2):
                    # print(padded_image[i+a, j+b])
                    # print(kernel[a+1,b+1])
                    conv_pixel += padded_image[i+a, j+b] * kernel[a+1, b+1]
            # conv_pixel = np.array(np.dot(image_array, mask_array), dtype=int)
            # print(conv_pixel)
            padded_image[i, j] = conv_pixel

    return padded_image[n_rows:image.shape[0]+n_rows, n_cols: image.shape[1] + n_cols]


def add_two_images(image1, image2):

    sum_image = np.zeros(image1.shape)

    try:
        if image1.size != image2.size:
            raise NameError()

        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):

                sum_image[i, j] = image1[i, j] + image2[i, j]

        return sum_image

    except NameError:
        print("Images don't have the same size")


def subtract_two_images(image1, image2):

    sub_image = np.zeros(image1.shape)

    try:
        if image1.size != image2.size:
            raise NameError()

        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):

                sub_image[i, j] = image1[i, j] - image2[i, j]

        return sub_image

    except NameError:
        print("Images don't have the same size")


def convolve_average(image, kernel):

    kernel_sum = np.dot(np.reshape(kernel, kernel.size), np.ones(kernel.size))

    average = (1/kernel_sum) * kernel

    average_im = convolve2d(image, average)
    return average_im


def convolve_percentil(image, kernel, percentil):

    per_func = lambda a, p: np.sort(a)[int(a.size*p)]

    new_rows = kernel.shape[0] - 1
    new_columns = kernel.shape[1] - 1
    # number of rows and columns from the center of the mask
    n_middle_r = new_rows // 2
    n_middle_c = new_columns // 2
    # create a new image with black borders
    padded_image = np.zeros((image.shape[0] + new_rows, image.shape[1] + new_columns))
    padded_image[n_middle_r: image.shape[0] + n_middle_r, n_middle_c: image.shape[1] + n_middle_c] = image

    for i in range(n_middle_r, image.shape[0] + n_middle_r):
        for j in range(n_middle_c, image.shape[1] + n_middle_c):

            image_crop = padded_image[i - n_middle_r: i + n_middle_r + 1, j - n_middle_c: j + n_middle_c + 1]
            image_crop_array = np.reshape(image_crop, image_crop.size)
            convoluted_pixel = per_func(image_crop_array, percentil)
            padded_image[i, j] = convoluted_pixel

    conv_image = padded_image[n_middle_r: image.shape[0] + n_middle_r, n_middle_c: image.shape[1] + n_middle_c]

    return conv_image


def convolve_laplace(image):

    laplace = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=float)

    im_laplace = convolve2d(image, laplace)
    # im_laplace = filters.laplace(image)
    # np.set_printoptions(suppress=True)
    # im_laplace = exposure.rescale_intensity(im_laplace, in_range=(0, 255))
    # for i in im_laplace:
    #     print(i)
    # im_laplace = add_two_images(image, im_laplace)
    return im_laplace


def convolve_sobel(image):

    sobel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    im_sobel = convolve2d(image, sobel_x)
    # im_sobel = exposure.rescale_intensity(im_sobel, out_range=(0, 2 ** 8 - 1))
    # im_sobel = filters.sobel(image)

    return im_sobel

def highboost_filter(original_image):

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],

    ], dtype=float)

    original_image = img_as_float(original_image)

    blured_image = convolve_average(original_image, kernel)
    mask = subtract_two_images(original_image, blured_image)
    highboost_image = add_two_images(original_image, mask)
    highboost_image = exposure.rescale_intensity(highboost_image, out_range=np.uint8)

    return highboost_image