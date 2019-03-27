import cv2 as cv
import numpy as np
from scipy.signal import convolve2d as conv2


class Part3(object):

    def deconvolution(self, blur_list, kerneld, kernelh, kernelv):

        for i in range(len(blur_list)): # for choosing the kernel
            if i == 0:
                kernel = kernelv
            if i == 1:
                kernel = kerneld
            if i == 2:
                kernel = kernelh

            fftk = np.fft.fft2(kernel)
            fftk = fftk + 0.00000001
            fftb = np.fft.fft2(blur_list[i])

            restored = fftb / fftk
            restored = np.fft.ifft2(restored)
            restored = np.fft.fftshift(restored)
            restored = np.abs(restored)

            cv.imwrite('../results/part3a/Deconvolved Image ' + str(i + 1) + ' without Regularization.jpg', restored)

    def withregularization(self, blur_list, kerneld, kernelh, kernelv, threshold):

        for i in range(len(blur_list)):    # for choosing the kernel
            if i == 0:
                kernel = kernelv
            if i == 1:
                kernel = kerneld
            if i == 2:
                kernel = kernelh

            fftb = (np.fft.fft2(blur_list[i]))
            fftk = (np.fft.fft2(kernel))
            fftb = np.where(np.abs(fftk) > threshold, fftb / (fftk + 0.000001), 0)
            restored = np.fft.ifft2(fftb)
            restored = np.fft.fftshift(restored)
            restored = np.abs(restored)

            cv.imwrite('../results/part3b/Deconvolved Image ' + str(i + 1) + ' with Regularization.jpg', restored)

    def bonus(self, blur_list, iterative):

        for x in range(len(blur_list)):
            row, col = np.shape(blur_list[x])
            crow, ccol = int(row / 2), int(col / 2)

            otf = np.zeros((row, col))    # actually otf is the fft of psf. but i create again
            otf[crow - 2:crow + 2, ccol - 2:ccol + 2] = 1 / 25   # because i got some error with conv2 func
            otf = np.fft.fft2(otf)          # so i changed size of otf(psf) after conv2 func
            i = np.ones((row, col)) / 2    # assigning 0.5 for I^0

            for y in range(0, iterative):
                ffti = np.fft.fft2(i)
                mul = otf * ffti
                imul = np.fft.ifft2(mul)
                div = blur_list[x] / imul
                fdiv = np.fft.fft2(div)
                res = otf * fdiv
                ires = np.fft.ifft2(res)
                i = ires * i

            cv.imwrite('../results/part3c/Restored Image ' + str(x) + ' with bonus.jpg', np.abs(np.fft.fftshift(i)))
