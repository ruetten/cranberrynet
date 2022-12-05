import cv2
import numpy as np
from matplotlib import pyplot as plt
from pykuwahara import kuwahara
import sys

##### DETECT CONTOURS #####
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# reading image
orig_img = cv2.imread('../Images/' + sys.argv[1])
#img = cv2.imread('../Images/cranberry_images/A5S19.jpg')
#img = cv2.imread('Images/Berries1-D.JPG')
img = rescaleFrame(orig_img, scale=0.5)

img = kuwahara(img, method='gaussian', radius=50)

edges = cv2.Canny(img,100,200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

i = 0
for contour in contours:
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area > 2300:
        cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

# list for storing names of shapes
i = 0
for contour in contours:
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)

    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # cv2.approxPloyDP() function to approximate the shape
    # approx = cv2.approxPolyDP(
    #     contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
    i = i + 1

# displaying the image after drawing contours
# cv2.imshow('Cranberries', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


###### DETECT LOCULES ######
img = rescaleFrame(orig_img, scale=0.5)
img = kuwahara(img, method='gaussian', radius=5)

new_image = np.zeros(img.shape, img.dtype)

alpha = 2.95 # Simple contrast control
beta = -500    # Simple brightness control
new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
img = kuwahara(img, method='gaussian', radius=5)

# i = 0
for contour in contours:
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area > 2300:
        cv2.drawContours(new_image, [hull], 0, (0, 255, 0), 2)
    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

new_image = 255 - new_image
th, new_image = cv2.threshold(new_image, 150, 255, cv2.THRESH_BINARY)

cv2.watershed()

cv2.imshow('Original Image', img)
cv2.imshow('New Image', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
