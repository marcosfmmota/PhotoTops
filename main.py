#main.py

import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math

def negative(image):

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 255 - image[i, j]

def log_transform(image, c = 1):

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = c * math.log(1 + image[i, j], 2)


def main():

    dir_name = "/home/marcosfe/Documents/PhotoTops/DIP3E_CH03"
    filename = input("Type the image name \n")
    filename = os.path.join(dir_name,filename)
    image = io.imread(filename)
    print(image.shape)
    log_transform(image)
    plt.imshow(image, cmap="gray")
    plt.show()
    negative(image)
    plt.imshow(image, cmap = "gray")
    plt.show()


if __name__ == "__main__":
    main()
