from skimage import io
from matplotlib import pyplot as plt
import color_models as cm
import color_filters as cf
from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
from skimage.util import img_as_float
import spatial_filters as sf
import numpy as np
import math


def test_color_models(image):
    fig = plt.figure("Color Model")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Modified Image")
    hsi_image = cm.rgb_to_hsi(image)
    rgb_image = cm.hsi_to_rgb(hsi_image)
    plt.imshow(rgb_image, cmap='gray')
    plt.show()


def test_brightness(image):
    fig = plt.figure("Color Model")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Modified Image")
    hsi_image = cf.brightness_filter(image, 0.3)
    plt.imshow(hsi_image, cmap='hsv')
    plt.show()


def test_average_filter(image):
    fig = plt.figure("Color Model")
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image)
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("RGB Image")
    avg_image = cf.average_filter_rgb(image)
    plt.imshow(avg_image, cmap='gray')
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("HSI Image")
    hsi_image = rgb2hsv(image)
    hsi_image = cf.average_filter_hsi(hsi_image)
    hsi_image = hsv2rgb(hsi_image)
    plt.imshow(hsi_image, cmap='gray')
    plt.show()


def test_tone_filter_rgb(image):
    fig = plt.figure("Color Model")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Modified Image")
    hsi_image = cf.tone_filter_rgb(img_as_float(image), lambda x: math.pow(x, 1.25))
    plt.imshow(hsi_image, cmap='hsv')
    plt.show()


def test_sepia(image):

    fig = plt.figure("Color Model")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Modified Image")
    hsi_image = cf.sepia_filter(img_as_float(image))
    plt.imshow(hsi_image, cmap='gray')
    plt.show()


def test_histogram(image):

    fig = plt.figure("Color Model")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Modified Image")
    hsi_image = cf.image_equalization(image)
    plt.imshow(hsi_image, cmap='gray')
    plt.show()

def test_chroma_key(image1, image2):
    chroma = cf.chroma_key(img_as_float(image1), img_as_float(image2))
    plt.imshow(chroma, cmap="gray")
    plt.show()

def test_batch_CH06():
    image = io.imread("dog.jpg")
    # test_color_models(image)
    # test_brightness(image)
    # test_average_filter(image)
    # test_tone_filter_rgb(image)
    # test_sepia(image)
    test_histogram(image)
    donald = io.imread("donald.jpg")
    back = io.imread("back.jpg")
    test_chroma_key(donald, back)
