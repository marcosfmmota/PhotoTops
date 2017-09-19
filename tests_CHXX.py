import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io

import spatial_filters as sf


def test_negative(dir_name, filename):

    filename = os.path.join(dir_name,filename)
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

    filename = os.path.join(dir_name,filename)
    image = io.imread(filename)
    fig = plt.figure("Original vs. Negative")
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    p_image = sf.power_transform(image, 1, 0.4)
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(p_image, cmap="gray")
    plt.show()

def test_histogram(dir_name, filename):

    filename = os.path.join(dir_name,filename)
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

    filename = os.path.join(dir_name,filename)
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
    plt.imshow(eq_image, cmap = "gray")
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
    plane = sf.bit_plane_slicing(image,[1, 1, 1, 1, 1, 1, 1, 1])
    plt.imshow(plane, cmap="gray")
    plt.show()

def test_batch_CH03():
    # dir_name = "/home/marcosfe/Documents/PhotoTops/DIP3E_CH03"
    dir_name = "C:\\Users\\MarcosFelipe\\Documents\\PhotoTops\\DIP3E_CH03"
    test_negative(dir_name,"Fig0304(a)(breast_digital_Xray).tif")
    test_logarithm(dir_name, "Fig0305(a)(DFT_no_log).tif")
    test_gamma(dir_name, "Fig0307(a)(intensity_ramp).tif")
    test_bit_plane_slicing(dir_name, "Fig0314(a)(100-dollars).tif")
    test_histogram(dir_name, "Fig0316(1)(top_left).tif")
    test_histogram(dir_name, "Fig0316(2)(2nd_from_top).tif")
    test_histogram_equalization(dir_name, "Fig0309(a)(washed_out_aerial_image).tif")
    test_histogram_equalization(dir_name, "Fig0316(2)(2nd_from_top).tif")
