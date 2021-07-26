import cv2
import numpy as np

def process_frame(diff, frame1):
    # convert image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur to image to remove noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply threshold to image to convert to binary image
    _, threshold = cv2.threshold(blur, 22, 255, cv2.THRESH_BINARY)

    # dialate threshold image to bring out edges more
    dialate = cv2.dilate(threshold, None, iterations=3)

    # find contours from dialated image
    contours, hierachy = cv2.findContours(dialate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # loop through all contours to find bounding box rectangle coordinates
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # calculate area of contour, to eliminate smaller contours
        if cv2.contourArea(contour) < 10000:
            continue

        # draw rectangle on image
        frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (21, 123, 98), 3)
    return frame1

# create video capture object
cap = cv2.VideoCapture('static.mp4')

# get the first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()


while cap.isOpened():
    # get the difference between current and previous frame
    diff = cv2.absdiff(frame1, frame2)

    # pass difference to process frame function for further processing
    result = process_frame(diff, frame1)

    # show processed result 
    cv2.imshow('Recorded Feed', result)

    frame1 = frame2

    _, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()