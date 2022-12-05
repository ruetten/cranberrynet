import cv2
import numpy as np
from matplotlib import pyplot as plt
from pykuwahara import kuwahara
import sys

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# reading image
filename = sys.argv[1]
input_dir = '../Images/'
output_dir = '../Output/'
input_filename = input_dir + filename
img = cv2.imread(input_filename)
#img = cv2.imread('../Images/cranberry_images/A5S19.jpg')
#img = cv2.imread('Images/Berries1-D.JPG')
img = rescaleFrame(img, scale=0.5)

img = kuwahara(img, method='gaussian', radius=50)

edges = cv2.Canny(img,100,200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

i = 0
myBlobs = []
for contour in contours:
    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area > 2300 and area < 10000:
        continueFlag = True

        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            center = (int(x), int(y))
            radius = 1
            cv2.circle(img, center, radius, (0,0,255),2)

            for blob in myBlobs:
                if x < blob[0]+blob[2] and x > blob[0] and y < blob[1]+blob[3] and y > blob[1]:
                    continueFlag = False

        if continueFlag:
            ellipse = cv2.fitEllipse(contour)
            #cv2.ellipse(img, ellipse, (0, 255, 0), 2)

            buffer = 10
            x,y,w,h = cv2.boundingRect(contour)
            myBlobs.append([x,y,w,h])
            cv2.rectangle(img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,255,0,2))
            cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)

            x,y,w,h = cv2.boundingRect(contour)
            cropped_img = img[y-buffer:y+h+buffer, x-buffer:x+w+buffer]

            filename_split = filename.split('.')
            output_filename = output_dir+filename_split[0]+'_'+str(i)+'.'+filename_split[1]
            #cv2.imwrite(output_filename,cropped_img)
            cv2.imshow(output_filename,cropped_img)
    i = i + 1

# displaying the image after drawing contours
cv2.imshow('Cranberries', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
