import os
from skimage import io
from skimage import img_as_float
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
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
    plt.imshow(image, cmap="gray")
    plt.show()
    hist = sf.histogram(image)
    sf.show_histogram(hist)

def test_histogram_equalization(dir_name, filename):
    test_histogram(dir_name, filename)
    filename = os.path.join(dir_name,filename)
    image = io.imread(filename)
    eq_image = sf.histogram_equalization(image)
    plt.imshow(eq_image, cmap = "gray")
    plt.show()
    hist = sf.histogram(eq_image)
    sf.show_histogram(hist)



def test_batch_CH03():
    dir_name = "/home/marcosfe/Documents/PhotoTops/DIP3E_CH03"
    test_negative(dir_name,"Fig0304(a)(breast_digital_Xray).tif")
    test_logarithm(dir_name, "Fig0305(a)(DFT_no_log).tif")
    test_gamma(dir_name, "Fig0307(a)(intensity_ramp).tif")
    test_histogram(dir_name, "Fig0316(1)(top_left).tif")
    test_histogram(dir_name, "Fig0316(2)(2nd_from_top).tif")
    test_histogram_equalization(dir_name, "Fig0316(1)(top_left).tif")
    test_histogram_equalization(dir_name, "Fig0316(2)(2nd_from_top).tif)
