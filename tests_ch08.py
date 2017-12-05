from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import compression as co



def test_compression():

    filename = "lena.bmp"
    image = io.imread(filename)
    filename = filename.split(".")[0]

    co.write_as_mcf(image, filename)

    image = co.read_as_np(filename, image.shape)
    t = np.array([84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, 265, 259, 261, 263])
    ct = co.lzw_decompress(t)
    w = ""
    for i in ct:
        w += chr(i)
    print(w)
    # plt.imshow(image, cmap="gray")
    # plt.show()

