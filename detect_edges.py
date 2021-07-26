import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect(image, kernel=(3, 3), threshold=[50, 50]):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur image to remove noise
    blur = cv2.GaussianBlur(gray, kernel, 0.3)

    # appy dialation to show edges more
    image_dialate = cv2.dilate(blur, kernel, iterations=3)

    # apply canny to detect edges
    canny = cv2.Canny(image_dialate, threshold1=threshold[0], threshold2=threshold[1])

    # apply laplacian to detect edges
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # return image
    return canny


# read in image
image = cv2.imread('toy.jpg', 1)

# process image
processed_image = detect(image)

# show original image and processed image

images = [image, processed_image]
titles = ['Original', 'Processed']

for i in range(len(images)):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
cv2.destroyAllWindows()
