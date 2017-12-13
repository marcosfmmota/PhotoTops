import numpy as np
import matplotlib.pyplot as plt
import color_models as cm
import spatial_filters as sf
from skimage import io
from skimage.util import pad


def compute_fft(image):
    padded_image = pad(image, ((0, image.shape[0]), (0, image.shape[1])), 'constant', constant_values=0)
    # print(padded_image.shape)
    fft_image = np.fft.fft2(padded_image)
    fft_image = np.fft.fftshift(fft_image)

    return fft_image


def invert_fft(fft_image):

    inv_fft = np.fft.ifftshift(fft_image)
    inv_fft = np.fft.ifft2(inv_fft)
    inv_fft = np.abs(inv_fft)
    image_filtered = inv_fft[:fft_image.shape[0] // 2, :fft_image.shape[1] // 2]

    return image_filtered


class FrequencyPicker(object):

    def __init__(self, ax, fft_img):

        self.ax = ax
        self.fft_img = fft_img
        self.cid1 = ax.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.lx = ax.axhline(color='r')
        self.xs = 0
        self.ys = 0
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def onmove(self, event):
        if event.inaxes != self.ax:
            return
        self.xs = int(event.xdata)
        self.ys = int(event.ydata)
        annot = self.ax.annotate("", xy=(self.xs, self.ys),  bbox=dict(boxstyle="round", fc="w"))
        annot.set_visible(True)
        self.lx.set_ydata(self.ys)
        self.txt.set_text('y=%d' % self.ys)
        # print('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()

    def onclick(self, event):
        # print("click ", event)
        if event.inaxes != self.ax: return
        self.xs = int(event.xdata)
        self.ys = int(event.ydata)
        print((self.xs, self.ys))
        spectrum = np.abs(self.fft_img)
        fig = plt.figure()
        n_ax = fig.add_subplot(111)
        n_ax.plot(spectrum[:, self.ys])
        fig.show()


def run():

    image = io.imread("lena.bmp")
    fig = plt.figure()
    ax = fig.add_subplot(121)
    image = cm.rgb_to_grayscale(image.astype(np.float))
    plt.imshow(image, cmap="gray")
    ax2 = fig.add_subplot(122)
    ax2.set_title("Escolha uma linha do espectro para ser alterada")
    fft_img = compute_fft(image)
    spectrum = np.abs(fft_img)
    sf.log_transform(spectrum)
    plt.imshow(spectrum, cmap="gray")
    freq_pick = FrequencyPicker(ax2, fft_img)

    plt.show()
