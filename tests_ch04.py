import math
import os

import matplotlib.pyplot as plt
from skimage import io

import frequency_filters as ff
import spatial_filters as sf


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
    ax.set_title("Spectrum")
    spectrum = ff.compute_spectrum(image)
    sf.log_transform(spectrum)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_low_pass_ideal_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Ideal lowpass")
    d0 = 60
    func = lambda duv, d0=d0: 1 if duv <= d0 else 0
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_low_pass_butterworth_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Butterworth lowpass")
    d0 = 60
    n = 1
    func = lambda duv, d0=d0, n=n: 1 / (1 + (duv / d0)**(2*n))
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_low_pass_gaussian_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Gaussian lowpass")
    d0 = 60
    func = lambda duv, d0=d0: math.exp(-(duv**2)/(2*(d0**2)))
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_high_pass_ideal_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Ideal Highpass")
    d0 = 60
    func = lambda duv, d0=d0: 0 if duv <= d0 else 1
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_high_pass_butterworth_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Butterworth highpass")
    d0 = 60
    n = 1
    func = lambda duv, d0=d0, n=n: 1 / (1 + (d0 / duv)**(2*n)) if duv != 0 else duv
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_high_pass_gaussian_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Gaussian highpass")
    d0 = 60
    func = lambda duv, d0=d0: 1 - math.exp(-(duv**2)/(2*(d0**2)))
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_bandreject_ideal_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Ideal Bandreject")
    d0 = 60
    W = 20
    func = lambda duv, d0=d0: 0 if duv <= (d0 - W/2) and duv <= (d0 + W/2) else 1
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_bandreject_butterworth_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Butterworth highpass")
    d0 = 60
    n = 1
    W = 20
    func = lambda duv, d0=d0, n=n: 1 / (1 + ((duv*W) / ((duv**2) - (d0**2)))**(2*n)) \
        if ((duv**2) - (d0**2)) != 0 else duv
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_bandreject_gaussian_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Gaussian bandreject")
    d0 = 60
    W = 10
    func = lambda duv, d0=d0: 1 - math.exp(-(duv**2 - d0**2)/(duv*W)) if duv != 0 else duv
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_bandpass_ideal_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Ideal Bandpass")
    d0 = 60
    W = 20
    func = lambda duv, d0=d0: 1 if duv <= (d0 - W/2) and duv <= (d0 + W/2) else 0
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_bandpass_butterworth_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Border detection")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Butterworth Bandpass")
    d0 = 60
    n = 1
    W = 20
    func = lambda duv, d0=d0, n=n: 1 - (1 / (1 + ((duv*W) / ((duv**2) - (d0**2)))**(2*n))) \
        if ((duv**2) - (d0**2)) != 0 else duv
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_bandpass_gaussian_filter(dirname, filename):

    filename = os.path.join(dirname, filename)
    image = io.imread(filename)
    fig = plt.figure("Frequency Filters")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Gaussian bandreject")
    d0 = 60
    W = 10
    func = lambda duv, d0=d0: math.exp(-(duv**2 - d0**2)/(duv*W)) if duv != 0 else duv
    spectrum = ff.general_frequency_filter(image, func)
    plt.imshow(spectrum, cmap="gray")
    plt.show()


def test_batch_CH04():

    dir_name = "C:\\Users\\MarcosFelipe\\Documents\\PhotoTops\\DIP3E_CH04"
    # test_shrink_image(dir_name, "Fig0417(a)(barbara).tif")
    # test_shrink_average_image(dir_name, "Fig0417(a)(barbara).tif")
    test_compute_spectrum(dir_name, "Fig0424(a)(rectangle).tif")
    test_compute_spectrum(dir_name, "Fig0427(a)(woman).tif")
    # test_low_pass_ideal_filter(dir_name, "Fig0441(a)(characters_test_pattern).tif")
    # test_low_pass_butterworth_filter(dir_name, "Fig0441(a)(characters_test_pattern).tif")
    # test_low_pass_gaussian_filter(dir_name, "Fig0441(a)(characters_test_pattern).tif")
    # test_high_pass_ideal_filter(dir_name, "Fig0441(a)(characters_test_pattern).tif")
    # test_high_pass_butterworth_filter(dir_name, "Fig0441(a)(characters_test_pattern).tif")
    # test_high_pass_gaussian_filter(dir_name, "Fig0441(a)(characters_test_pattern).tif")
    # test_bandreject_ideal_filter(dir_name, "Fig0462(a)(PET_image).tif")
    # test_bandreject_butterworth_filter(dir_name, "Fig0462(a)(PET_image).tif")
    # test_bandreject_gaussian_filter(dir_name, "Fig0462(a)(PET_image).tif")
    # test_bandpass_ideal_filter(dir_name, "Fig0462(a)(PET_image).tif")
    # test_bandpass_butterworth_filter(dir_name, "Fig0462(a)(PET_image).tif")
    # test_bandpass_gaussian_filter(dir_name, "Fig0462(a)(PET_image).tif")
