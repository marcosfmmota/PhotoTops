import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from skimage.util import img_as_float
import spatial_filters as sf
import frequency_filters as ff


def test_negative(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Original vs. Negative")
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    sf.negative(image)
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image, cmap="gray")
    plt.show()


def test_logarithm(dir_name, filename):

    filename = os.path.join(dir_name,filename)
    image = io.imread(filename)
    fig = plt.figure("Original vs. Negative")
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    sf.log_transform(image)
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image, cmap="gray")
    plt.show()


def test_gamma(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Original vs. Negative")
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    p_image = sf.power_transform(image, 1, 0.4)
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(p_image, cmap="gray")
    plt.show()


def test_histogram(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Image vs Histogram")
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    hist = sf.histogram(image)
    x_pos = np.arange(len(hist))
    plt.bar(x_pos, hist, width=1.0)
    plt.show()


def test_histogram_equalization(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Histogram Equalization")
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Original Histogram")
    hist = sf.histogram(image)
    x_pos = np.arange(len(hist))
    plt.bar(x_pos, hist, width=1.0)
    eq_image = sf.histogram_equalization(image)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Equalized Image")
    plt.imshow(eq_image, cmap="gray")
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Equalized Histogram")
    hist = sf.histogram(image)
    x_pos = np.arange(len(hist))
    plt.bar(x_pos, hist, width=1.0)
    plt.show()


def test_bit_plane_slicing(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Bit Plane Slicing")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Bit Plane")
    plane = sf.bit_plane_slicing(image, [0, 0, 0, 0, 1, 1, 1, 1])
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_contrast_stretching(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Contrast Streching")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Contrast enhanced image")
    sf.contrast_stretching(image, (128, 55), (168, 255))
    plt.imshow(image, cmap="gray")
    plt.show()


def test_local_histogram_equalization(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Local Histogram Equalization")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Equalized Image")
    sf.local_equalization(image)
    plt.imshow(image, cmap="gray")
    plt.show()


def test_average(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Blur filter")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Blured")
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    plane = sf.convolve_average(image, kernel)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_median(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Blur filter")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Blured")
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    plane = sf.convolve_median(image, kernel)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_laplace(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Laplace")
    plane = sf.convolve_laplace(image)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_highboost(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("HighBoost Image")
    plane = sf.highboost_filter(image)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_sobel_x(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Sobel")
    plane = sf.convolve_sobel_x(image)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_sobel_y(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Sobel")
    plane = sf.convolve_sobel_y(image)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_enhanced_borders_sobel(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Sobel")
    plane = sf.enhance_borders_sobel(image)
    plt.imshow(plane, cmap="gray")
    plt.show()


def test_batch_CH03():
    # dir_name = "/home/marcosfe/Documents/PhotoTops/DIP3E_CH03"
    dir_name = "C:\\Users\\MarcosFelipe\\Documents\\PhotoTops\\DIP3E_CH03"
    # test_negative(dir_name,"Fig0304(a)(breast_digital_Xray).tif")
    # test_logarithm(dir_name, "Fig0305(a)(DFT_no_log).tif")
    # test_gamma(dir_name, "Fig0307(a)(intensity_ramp).tif")
    # test_bit_plane_slicing(dir_name, "Fig0314(a)(100-dollars).tif")
    # test_contrast_stretching(dir_name, "Fig0312(a)(kidney).tif")
    # test_histogram(dir_name, "Fig0316(1)(top_left).tif")
    # test_histogram_equalization(dir_name, "Fig0309(a)(washed_out_aerial_image).tif")
    test_local_histogram_equalization(dir_name, "Fig0326(a)(embedded_square_noisy_512).tif")
    # test_average(dir_name, "Fig0333(a)(test_pattern_blurring_orig).tif")
    # test_median(dir_name, "Fig0335(a)(ckt_board_saltpep_prob_pt05).tif")
    # test_highboost(dir_name, "Fig0340(a)(dipxe_text).tif")
    # test_laplace(dir_name, "Fig0338(a)(blurry_moon).tif")
    # test_sobel_x(dir_name, "Fig0338(a)(blurry_moon).tif")
    # test_sobel_y(dir_name, "Fig0338(a)(blurry_moon).tif")
    # test_enhanced_borders_sobel(dir_name, "Fig0338(a)(blurry_moon).tif")


def test_shrink_image(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Shrinked Image")
    shrink = ff.shrink_image(image, 2)
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_shrink_average_image(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Shrinked Image")
    shrink = ff.shrink_average_image(image, 2)
    plt.imshow(shrink, cmap="gray")
    plt.show()

def test_compute_spectrum(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Shrinked Image")
    spectrum = ff.compute_spectrum(image)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_batch_CH04():

    dir_name = "C:\\Users\\MarcosFelipe\\Documents\\PhotoTops\\DIP3E_CH04"
    test_shrink_image(dir_name, "Fig0417(a)(barbara).tif")
    test_shrink_average_image(dir_name, "Fig0417(a)(barbara).tif")
    test_compute_spectrum(dir_name, "Fig0424(a)(rectangle).tif")
    test_compute_spectrum(dir_name, "Fig0427(a)(woman).tif")

