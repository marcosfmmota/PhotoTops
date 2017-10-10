# tests of restoration techniques in the samples images of chapter 05
from skimage import io
import os
import matplotlib.pyplot as plt
import restoration_filters as rf
import numpy as np


def test_arithmetic_mean(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Mean Image")
    shrink = rf.arithmetic_mean_filter(image.astype(np.float))
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_geometric_mean(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Geometric Mean Image")
    shrink = rf.geometric_mean_filter(image.astype(np.float))
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_harmonic_mean(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Mean Image")
    shrink = rf.harmonic_mean_filter(image)
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_contraharmonic_mean(dir_name, filename1, filename2):

    filename = os.path.join(dir_name, filename1)
    image1 = io.imread(filename)
    filename = os.path.join(dir_name, filename2)
    image2 = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Pepper Noise Image")
    plt.imshow(image1, cmap="gray")
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Salt Noise Image")
    plt.imshow(image2, cmap="gray")
    rf.contraharmonic_mean_filter(image1.astype(np.float), (3, 3), 1.5)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Pepper Noise Image Restored")
    plt.imshow(image1, cmap="gray")
    rf.contraharmonic_mean_filter(image2.astype(np.float), (3, 3), -1.5)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Salt Noise Image Restored")
    plt.imshow(image2, cmap="gray")
    plt.show()


def test_max(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Max Image")
    shrink = rf.max_filter(image)
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_min(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Min Image")
    shrink = rf.min_filter(image)
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_alpha_trimmed(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Alpha Image")
    shrink = rf.alpha_trimmed_filter(image.astype(np.float), 5, (5, 5))
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_mean_adaptive(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Alpha Image")
    shrink = rf.mean_adaptive_filter(image.astype(np.float), 1000, (7, 7))
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_median_adaptive(dir_name, filename):

    filename = os.path.join(dir_name, filename)
    image = io.imread(filename)
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(image, cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Median Adaptive Image")
    shrink = rf.median_adaptive_filter(image.astype(np.float), (7, 7))
    plt.imshow(shrink, cmap="gray")
    plt.show()


def test_batch_CH05():

    dir_name = "C:\\Users\\MarcosFelipe\\Documents\\PhotoTops\\DIP3E_CH05"
    # test_arithmetic_mean(dir_name, "Fig0507(b)(ckt-board-gauss-var-400).tif")
    # test_geometric_mean(dir_name, "Fig0507(b)(ckt-board-gauss-var-400).tif")
    # test_harmonic_mean(dir_name, "Fig0508(b)(circuit-board-salt-prob-pt1).tif")
    # test_contraharmonic_mean(dir_name, "Fig0508(a)(circuit-board-pepper-prob-pt1).tif",
    #                          "Fig0508(b)(circuit-board-salt-prob-pt1).tif")
    # test_max(dir_name, "Fig0508(a)(circuit-board-pepper-prob-pt1).tif")
    # test_min(dir_name, "Fig0508(b)(circuit-board-salt-prob-pt1).tif")
    # test_alpha_trimmed(dir_name, "Fig0512(a)(ckt-uniform-var-800).tif")
    # test_mean_adaptive(dir_name, "Fig0513(a)(ckt_gaussian_var_1000_mean_0).tif")
    test_median_adaptive(dir_name, "Fig0514(a)(ckt_saltpep_prob_pt25).tif")

