import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Part2(object):

    def findingblurkernels(self, blur_list):

        rows, cols = np.shape(blur_list[0])   # kernel creating
        crow, ccol = int(rows/2), int(cols/2)
        kerneld = np.zeros((rows, cols))
        kernelh = np.zeros((rows, cols))
        kernelv = np.zeros((rows, cols))

        for i in range(-10, 11):
            kerneld[crow+i, ccol+i] = 1/21
            kernelv[crow, ccol+i] = 1/21

        for i in range(-7, 8):
            kernelh[crow+i, ccol] = 1/15

        # this code block is for Examine the Magnitude Spec of images

        # for blur in blur_list:  # this is for Examine the FFT of all images
        #     fft = np.fft.fft2(blur)
        #     fshift = np.fft.fftshift(fft)
        #     magnitude_spectrum = 20 * np.log(np.abs(fshift))
        #     plt.imshow(magnitude_spectrum, cmap='gray'),
        #     plt.title('Magnitude Spectrum of part2 blur Image'), plt.xticks([]), plt.yticks([]),
        #     plt.show()

        return kerneld, kernelh, kernelv

    def applyonimage(self, img, kernelv, kerneld, kernelh):

        for i in range(0, 3):  # for choosing the kernel
            if i == 0:
                kernel = kernelv
            if i == 1:
                kernel = kerneld
            if i == 2:
                kernel = kernelh

            fftp = np.fft.fft2(img)
            fftk = np.fft.fft2(kernel)
            blurred = fftp * fftk
            restored = np.fft.ifft2(blurred)
            restored = np.fft.fftshift(restored)
            restored = np.abs(restored)
            cv.imwrite('../results/part2/Blurred Example Image with Found '+ str(i) + ' Kernel.jpg', restored)  # saving image

        # this code block is for Examine the Magnitude Spec of images

        # fshift = np.fft.fftshift(blurred)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # plt.imshow(magnitude_spectrum, cmap='gray'),
        # plt.title('Magnitude Spectrum of part2 Example Image'), plt.xticks([]), plt.yticks([]),
        # plt.show()

