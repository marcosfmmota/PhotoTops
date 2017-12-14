from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import compression as co



def test_compression():

    filename = "lena.bmp"
    image = io.imread(filename)
    filename = filename.split(".")[0]

    co.write_as_mcf(image, filename)

    vec = co.read_as_np(filename)
    com = co.compress(image)

    co.write_as_mcf(com, "comp_lena")
    img2 = co.read_as_np("comp_lena")
    print(img2.shape)
    image = co.decompress(img2, image.shape)
    plt.imshow(image, cmap="gray")
    plt.show()

