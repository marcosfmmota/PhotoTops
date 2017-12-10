from skimage import io
import morphological_operators as mo
import matplotlib.pyplot as plt
import color_models
import numpy as np


def test_erosion(filename):
    image = io.imread(filename)
    image = color_models.rgb_to_grayscale(image.astype(np.float))
    fig = plt.figure("Original vs Erosion")
    fig.add_subplot(121)
    plt.imshow(image, cmap="gray")
    mo.erosion(image)
    fig.add_subplot(122)
    plt.imshow(image, cmap="gray")
    plt.show()



def test_dilation(filename):
    image = io.imread(filename)
    image = color_models.rgb_to_grayscale(image.astype(np.float))
    fig = plt.figure("Original vs Dilation")
    fig.add_subplot(121)
    plt.imshow(image, cmap="gray")
    mo.dilation(image)
    fig.add_subplot(122)
    plt.imshow(image, cmap="gray")
    plt.show()


def test_morphological_gradient(filename):
    image = io.imread(filename)
    image = color_models.rgb_to_grayscale(image.astype(np.float))
    fig = plt.figure("Original vs Dilation")
    fig.add_subplot(121)
    plt.imshow(image, cmap="gray")
    img_grad = mo.morphological_gradient(image)
    fig.add_subplot(122)
    plt.imshow(img_grad, cmap="gray")
    plt.show()


def test_batch_CH09():

    filename = "lena.bmp"

    test_erosion(filename)
    test_dilation(filename)
    test_morphological_gradient(filename)