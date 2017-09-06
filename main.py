#main.py

import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def negative(image):

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 255 - image[i, j]


def main():

    dir_name = "/home/marcosfe/Documents/PhotoTops/DIP3E_CH03"
    filename = input("Type the image name \n")
    filename = os.path.join(dir_name,filename)
    image = io.imread(filename)
    print(image.shape)
    negative(image)
    plt.imshow(image, cmap = "gray")
    plt.show()


if __name__ == "__main__":
    main()
