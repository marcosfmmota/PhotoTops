#main.py
import os
from skimage import io
from skimage import img_as_float
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
import spatial_filters as sf
from tests_CHXX import test_batch_CH03

def main():
    #test_batch_CH03()
    dir_name = "/home/marcosfe/Documents/PhotoTops/DIP3E_CH03"
    filename = "Fig0326(a)(embedded_square_noisy_512).tif"
    filename = os.path.join(dir_name,filename)
    image = io.imread(filename)
    plt.imshow(image, cmap="gray")
    plt.show()
    image = sf.local_equalization(image)
    plt.imshow(image, cmap="gray")
    plt.show()
    
if __name__ == "__main__":
    main()
