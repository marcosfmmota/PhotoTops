# main.py
from tests_ch03 import test_batch_CH03
from tests_ch04 import test_batch_CH04
from tests_ch05 import test_batch_CH05

from skimage import io
from matplotlib import pyplot as plt
import color_models as cm

def main():

    # test_batch_CH03()
    # test_batch_CH04()
    # test_batch_CH05()
    image = io.imread("dog.jpg")
    fig = plt.figure("Restoration")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    print(image.shape)
    plt.imshow(image)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Median Adaptive Image")
    shrink = cm.rgb_to_hsi(image)
    plt.imshow(shrink, cmap='hsv')
    plt.show()


if __name__ == "__main__":
    main()
