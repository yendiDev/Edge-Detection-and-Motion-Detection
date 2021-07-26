import cv2
import numpy as np
import matplotlib.pyplot as plt

# create video reader object
cap = cv2.VideoCapture('static.mp4')

# get first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

print('Length of frame1 is: ', len(frame1))
print('Length of frame2 is: ', len(frame2))

print('Shape of frame 1: ', (frame1.shape))
print('Shape of frame 2: ', (frame2.shape))

images = [frame1, frame2]
titles = ['Frame1', 'Frame2']

# for i in range(len(images)):
#     plt.subplot(1, 2, i+1), plt.imshow(images[i])
#     plt.title(titles[i])

# plt.show()



while cap.isOpened():
    # get difference between first and second frame
    diff = cv2.absdiff(frame1, frame2)

    # convert difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # apply blur to frame to remove noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply threshold to image
    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # dialate image to improve contours finding
    dialate = cv2.dilate(threshold, None, iterations=3)

    # find contours in image difference
    contours, hierachy = cv2.findContours(dialate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # iterate through all contours to draw rectangle
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # calculate area, ignore smaller area
        if cv2.contourArea(contour) < 10000:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, 'Status: {}'.format('Movement'), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    # draw contours on image
    # frame1 = cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
 
    cv2.imshow('Feed', frame1)

    # assign old frame to previous frame
    frame1 = frame2

    # read in new frame
    _, frame2 = cap.read()

    print('len of frame1 ', len(frame1))
    print('Length of frame 2 ', len(frame2))

    # stop video when q is pressed
    if cv2.waitKey(40) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()