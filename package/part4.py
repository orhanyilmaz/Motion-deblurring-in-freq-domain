import numpy as np
import cv2 as cv
from scipy.signal import convolve2d as conv2
from skimage import color, data


class Part4(object):

    def myrichardsonlucy(self, iterative):

        astro = color.rgb2gray(data.astronaut())
        row, col = np.shape(astro)
        crow, ccol = int(row / 2), int(col / 2)

        psf = np.ones((5, 5)) / 25
        astro = conv2(astro, psf, 'same')
        # Add Noise to Image
        astro_noisy = astro.copy()
        astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

        otf = np.zeros((row, col))      # actually otf is the fft of psf. but i create again
        otf[crow - 2:crow + 2, ccol - 2:ccol + 2] = 1 / 25  # because i got some error with conv2 func
        otf = np.fft.fft2(otf)           # so i changed size of otf(psf) after conv2 func
        i = np.ones((row, col)) / 2         # assigning 0.5 for I^0

        for x in range(0, iterative):
            ffti = np.fft.fft2(i)
            mul = otf * ffti
            imul = np.fft.ifft2(mul)
            div = astro_noisy / imul
            fdiv = np.fft.fft2(div)
            res = otf * fdiv
            ires = np.fft.ifft2(res)
            i = ires * i

        cv.imwrite('../results/part4/Restored Image over Iteration.jpg', np.abs(np.fft.fftshift(i))*255)
