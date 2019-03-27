from part1 import Part1
from part2 import Part2
from part3 import Part3
from part4 import Part4
import os
import glob
import cv2 as cv
part1 = Part1()
part2 = Part2()
part3 = Part3()
part4 = Part4()

blur_list = []
blur_list2 = []

try:  # creating results folders
    if not os.path.exists('../results/part1/'):
        os.makedirs('../results/part1/')
    if not os.path.exists('../results/part2/'):
        os.makedirs('../results/part2/')
    if not os.path.exists('../results/part3a/'):
        os.makedirs('../results/part3a/')
    if not os.path.exists('../results/part3b/'):
        os.makedirs('../results/part3b/')
    if not os.path.exists('../results/part3c/'):
        os.makedirs('../results/part3c/')
    if not os.path.exists('../results/part4/'):
        os.makedirs('../results/part4/')
except OSError:
    print('Error')

for filename in sorted(glob.glob('../part1/*.png')):  # reading images
    im = cv.imread(filename, 0)
    blur_list.append(im)  # adding to list

for filename in sorted(glob.glob('../part2/*.png')):  # reading images
    im = cv.imread(filename, 0)
    blur_list2.append(im)  # adding to list

original = cv.imread('../part1/original.jpg', 0)
example = cv.imread('../part2/example_image.jpg', 0)

part1.exploringdeconvolution(blur_list, original)
kerneld, kernelh, kernelv = part2.findingblurkernels(blur_list2)
part2.applyonimage(example, kernelv, kerneld, kernelh)
part3.deconvolution(blur_list2, kerneld, kernelh, kernelv)
part3.withregularization(blur_list2, kerneld, kernelh, kernelv, 0.01)
part3.bonus(blur_list2, 15)
part4.myrichardsonlucy(15)

