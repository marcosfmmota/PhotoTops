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
        self.ifft_img = None
        self.cid1 = ax.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.lx = ax.axhline(color='r')
        self.lx2 = None
        self.ly2 = None
        self.n_ax = None
        self.x = 0
        self.y = 0
        self.xs = 0
        self.ys = 0.0
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
        self.txt2 = None
        self.fx = -1.0
        self.fyr = -1.0

    def onmove(self, event):
        if event.inaxes != self.ax:
            return
        self.x = int(event.xdata)
        self.y = int(event.ydata)
        self.lx.set_ydata(self.y)
        self.txt.set_text('y=%d' % self.y)
        # print('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()

    def onclick(self, event):
        # print("click ", event)
        if event.inaxes != self.ax: return
        self.x = int(event.xdata)
        self.y = int(event.ydata)
        print((self.x, self.y))
        spectrum = np.abs(self.fft_img)
        figi = plt.figure()
        n_ax = figi.add_subplot(111)
        n_ax.plot(spectrum[self.y, :])
        self.lx2 = n_ax.axhline(color='r')
        self.ly2 = n_ax.axvline(color='r')
        self.txt2 = n_ax.text(0.7, 0.9, '', transform=n_ax.transAxes)
        self.n_ax = n_ax
        figi.show()
        cidclose = figi.canvas.mpl_connect('close_event', self.pick_onclose)
        cidclick = figi.canvas.mpl_connect('button_press_event', self.pick_onclick)
        cidrelease = figi.canvas.mpl_connect('button_release_event', self.pick_onrelease)
        cidmove = figi.canvas.mpl_connect('motion_notify_event', self.pick_onmove)


    def pick_onmove(self, event):
        if event.inaxes != self.n_ax:
            return
        self.xs = int(event.xdata)
        self.ys = event.ydata
        self.lx2.set_ydata(self.ys)
        if self.fx != -1.0:
            self.ly2.set_xdata(self.fx)
            self.txt2.set_text('x=%d y=%d' % (self.fx, self.ys))
        else:
            self.ly2.set_xdata(self.xs)
            self.txt2.set_text('x=%d y=%d' % (self.xs, self.ys))
        # print('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()

    def pick_onclose(self, event):
        f_image = invert_fft(self.fft_img)
        self.ifft_img = f_image
        self.ax.imshow(f_image, cmap="gray")

    def pick_onclick(self, event):
        self.fx = int(event.xdata)
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

    def pick_onrelease(self, event):
        print("Release")
        self.fyr = event.ydata
        self.fft_img[self.y, self.xs] = self.fyr
        self.fft_img[self.y, -self.xs] = self.fyr
        self.fft_img[-self.y, self.xs] = self.fyr
        self.fft_img[-self.y, -self.xs] = self.fyr
        print(self.fyr)
        spectrum = np.abs(self.fft_img)
        self.n_ax.plot(spectrum[self.y, :])
        # self.n_ax.figure.show()
        self.fx = -1.0

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid1)
        self.ax.figure.canvas.mpl_disconnect(self.cid2)



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
    fig3 = plt.figure()
    ax3 = fig.add_subplot(111)
    dif_img = image - freq_pick.ifft_img
    plt.imshow(dif_img, cmap='gray')
    plt.show()
