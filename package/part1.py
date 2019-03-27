import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Part1(object):
    def exploringdeconvolution(self, blur_list, original):

        fft_im = np.fft.fft2(original)
        fft_blur = np.fft.fft2(blur_list[0])  # for examine , choose one of blur images
        kernel = fft_blur / fft_im
        restored = fft_blur / kernel
        restored = np.fft.ifft2(restored)
        restored = np.abs(restored)
        cv.imwrite('../results/part1/Restored Image.jpg', restored)  # saving image

        # this code block is for Examine the Magnitude Spec of images

        # fft = np.fft.fft2(original)
        # fshift = np.fft.fftshift(fft)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # plt.imshow(magnitude_spectrum, cmap='gray'),
        # plt.title('Magnitude Spectrum of part1 Original Image'), plt.xticks([]), plt.yticks([]),
        # plt.show()
        #
        # for blur in blur_list:
        #     fft = np.fft.fft2(blur)
        #     fshift = np.fft.fftshift(fft)
        #     magnitude_spectrum = 20 * np.log(np.abs(fshift))
        #     plt.imshow(magnitude_spectrum, cmap='gray'),
        #     plt.title('Magnitude Spectrum of part1 blur Images'), plt.xticks([]), plt.yticks([]),
        #     plt.show()
        #
        # fft = np.fft.fft2(restored)
        # fshift = np.fft.fftshift(fft)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # plt.imshow(magnitude_spectrum, cmap='gray'),
        # plt.title('Magnitude Spectrum of part1 Restored Image'), plt.xticks([]), plt.yticks([]),
        # plt.show()

