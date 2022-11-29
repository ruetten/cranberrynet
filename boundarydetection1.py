import cv2
import numpy as np
from matplotlib import pyplot as plt

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# reading image
img = cv2.imread('../Images/cranberry_images/A4S329.jpg')
img = rescaleFrame(img, scale=0.5)
img = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)

# converting image into grayscale image
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow('Cranberries', im_bw)

# thresh = 120
# im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('Cranberries', im_bw)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# setting threshold of gray image
thresh_val = 95
_, threshold = cv2.threshold(im_gray, thresh_val, 255, cv2.THRESH_BINARY)

# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# figure out which are cranberries and which are bordering circles
i = 0
for contour in contours:
    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    area = cv2.contourArea(contour)
    i = i + 1

# list for storing names of shapes
i = 0
totalCranberryArea = 0
for contour in contours:
    area = cv2.contourArea(contour)

    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # cv2.approxPloyDP() function to approximate the shape
    # approx = cv2.approxPolyDP(
    #     contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 10)

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    i = i + 1

# displaying the image after drawing contours
cv2.imshow('Cranberries', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
