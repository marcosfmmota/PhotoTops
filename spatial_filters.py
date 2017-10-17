import math

import numpy as np
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage import util
from skimage import exposure

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

    padded_image = util.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=0)

    for i in range(1, shape[0] + 1):
        for j in range(1, shape[1] + 1):

            grid_image = padded_image[i-1:i+2, j-1:j+2]

            trans_func = histogram(grid_image)
            acum = 0
            mn = grid_image.size

            for x in range(len(trans_func)):
                acum += trans_func[x]
                trans_func[x] = (255 / mn) * acum

            trans_func = np.round(trans_func)

            # for i in range(image.shape[0]):
            #     for j in range(image.shape[1]):
            # print(trans_func[image[i-1, j-1]])
            image[i-1, j-1] = trans_func[image[i-1, j-1]]

    return image


def convolve2d(image, kernel):

    kernel = np.fliplr(kernel)
    n_rows = kernel.shape[0] // 2
    n_cols = kernel.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0]+n_rows):
        for j in range(n_cols, image.shape[1]+n_cols):
            # print(padded_image[i, j])
            # print("#")
            conv_pixel = 0.0
            for a in range(-n_rows, n_rows+1):
                for b in range(-n_cols, n_cols+1):
                    aux_pix = padded_image[i+a, j+b]
                    aux_kernel = kernel[a+n_rows, b+n_cols]
                    conv_pixel += aux_pix * aux_kernel

            # print("*")
            # print(image_crop)
            # print(conv_pixel)
            # padded_image[i, j] = conv_pixel
            image[i - n_rows, j - n_cols] = conv_pixel
            # print(image[i - n_rows, j - n_cols])

    # return padded_image[n_rows:image.shape[0]+n_rows, n_cols: image.shape[1] + n_cols]
    return image


def add_two_images(image1, image2):

    sum_image = np.zeros_like(image1)

    try:
        if image1.size != image2.size:
            raise NameError()

        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):

                sum_pixel = (image1[i, j] + image2[i, j]) / 2.0
                if sum_pixel > 255:
                    sum_image[i, j] = 255
                else:
                    sum_image[i, j] = image1[i, j] + image2[i, j]

        return sum_image

    except NameError:
        print("Images don't have the same size")


def subtract_two_images(image1, image2):

    sub_image = np.empty_like(image1)

    try:
        if image1.size != image2.size:
            raise NameError()

        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):

                sub_pixel = (image1[i, j] - image2[i, j]) / 2.0
                if sub_pixel < 0:
                    sub_image[i, j] = 0
                else:
                    sub_image[i, j] = image1[i, j] - image2[i, j]

        return sub_image

    except NameError:
        print("Images don't have the same size")


def convolve_average(image, kernel):

    kernel_sum = np.dot(np.reshape(kernel, kernel.size), np.ones(kernel.size))

    average = (1/kernel_sum) * kernel

    average_im = convolve2d(image, average)
    return average_im


def convolve_median(image, kernel):

    per_func = lambda a: np.sort(a)[a.size//2]
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
            convoluted_pixel = per_func(image_crop_array)
            padded_image[i, j] = convoluted_pixel

    conv_image = padded_image[n_middle_r: image.shape[0] + n_middle_r, n_middle_c: image.shape[1] + n_middle_c]

    return conv_image


def convolve_laplace(image):

    laplace = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float)
    image_laplace = image.astype(np.float)

    convolve2d(image_laplace, laplace)

    return image_laplace


def convolve_sobel_x(image):

    sobel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float)
    im_sobel_x = image.astype(np.float)

    convolve2d(im_sobel_x, sobel_x)

    return im_sobel_x


def convolve_sobel_y(image):
    sobel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float)
    im_sobel_y = image.astype(np.float)

    convolve2d(im_sobel_y, sobel_y)

    return im_sobel_y


def enhance_borders_sobel(image):

    image_float = image.astype(np.float)

    borders_x = convolve_sobel_x(image_float)
    borders_y = convolve_sobel_y(image_float)

    enhanced_x = add_two_images(image.astype(np.float), borders_x)
    enhanced_y = add_two_images(image.astype(np.float), borders_y)

    final_image = add_two_images(enhanced_x, enhanced_y)
    return final_image


def highboost_filter(original_image):

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=float)

    float_image = original_image.astype(np.float)
    blured_image = convolve_average(float_image, kernel)
    mask = subtract_two_images(original_image.astype(np.float), blured_image)
    highboost_image = add_two_images(original_image, mask)

    return highboost_image