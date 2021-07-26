import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_contours(image):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply threshold to image
    _, threshold = cv2.threshold(gray, 127, 255, 0)

    # detect contours
    contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours

def draw_contours(image, contours, index=-1, color=(2, 222, 56), thickness=3):
    image = cv2.drawContours(image, contours, index, color, thickness)
    return image

# load image
image = cv2.imread('toy.jpg', 1)

# get contours
contours = detect_contours(image) 

# draw contours
contours_drawn = draw_contours(image, contours=contours)

# show contours
images = [image, contours_drawn]
titles = ['Original', 'Contours']

for i in range(len(images)):
    plt.subplot(1, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
print('The number of contours detected: ', len(contours))

