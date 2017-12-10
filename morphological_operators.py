import numpy as np
from skimage import util


def erosion(image: np.ndarray, se=np.ones((3, 3))):

    n_rows = se.shape[0] // 2
    n_cols = se.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0]+n_rows):
        for j in range(n_cols, image.shape[1]+n_cols):

            # for a in range(-n_rows, n_rows+1):
            #     for b in range(-n_cols, n_cols+1):
            #         aux_pix = padded_image[i+a, j+b]
            #         aux_kernel = se[a+n_rows, b+n_cols]
            #         conv_pixel += aux_pix * aux_kernel
            se_img = padded_image[i - n_rows: i + n_rows + 1, j - n_cols: j + n_cols + 1]
            # print(se_img.shape)
            mask = (se_img * se) > 0
            # print(se_img)
            mini = np.min(se_img[mask])

            image[i - n_rows, j - n_cols] = mini

    return image


def dilation(image: np.ndarray, se=np.ones((3, 3))):

    se = np.fliplr(se)
    n_rows = se.shape[0] // 2
    n_cols = se.shape[1] // 2
    padded_image = util.pad(image, ((n_rows, n_rows), (n_cols, n_cols)), 'constant', constant_values=0)

    for i in range(n_rows, image.shape[0] + n_rows):
        for j in range(n_cols, image.shape[1] + n_cols):
            # for a in range(-n_rows, n_rows+1):
            #     for b in range(-n_cols, n_cols+1):
            #         aux_pix = padded_image[i+a, j+b]
            #         aux_kernel = se[a+n_rows, b+n_cols]
            #         conv_pixel += aux_pix * aux_kernel
            se_img = padded_image[i - n_rows: i + n_rows + 1, j - n_cols : j + n_cols +1]
            # print(se_img.shape)
            mask = (se_img * se) > 0
            mini = np.max(se_img[mask])

            image[i - n_rows, j - n_cols] = mini

    return image


def morphological_gradient(image: np.ndarray):

    img1 = image.copy()
    img2 = image.copy()

    img1 = dilation(img1)
    img2 = erosion(img2)

    return img1 - img2
