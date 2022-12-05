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
orig_img = cv2.imread(input_filename)
#img = cv2.imread('../Images/cranberry_images/A5S19.jpg')
#img = cv2.imread('Images/Berries1-D.JPG')
rescaled_img = rescaleFrame(orig_img, scale=0.5)
img = rescaled_img

img = kuwahara(img, method='gaussian', radius=50)

edges = cv2.Canny(img,100,200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

i = -1
myBlobs = []
myCrans = []
for contour in contours:
    # here we are ignoring first contour because
    # findcontour function detects whole image as shape
    if i == -1:
        i = 0
        continue

    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area > 2300 and area < 18000:
        continueFlag = True

        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            cent_x = int(M['m10']/M['m00'])
            cent_y = int(M['m01']/M['m00'])
            center = (int(cent_x), int(cent_y))
            radius = 1

            for blob in myBlobs:
                if cent_x < blob[0]+blob[2] and cent_x > blob[0] and cent_y < blob[1]+blob[3] and cent_y > blob[1]:
                    continueFlag = False

        if continueFlag:
            x,y,w,h = cv2.boundingRect(contour)
            myBlobs.append([x,y,w,h])

            buffer = 0
            cropped_img = rescaled_img[y-buffer:y+h+buffer, x-buffer:x+w+buffer]
            myCrans.append(cropped_img)

            filename_split = filename.split('.')
            output_filename = output_dir+filename_split[0]+'_'+str(i)+'.'+filename_split[1]
            cv2.imwrite(output_filename,cropped_img)
            #cv2.imshow(output_filename,cropped_img)

            cv2.circle(img, center, radius, (0,0,255),2)

            ellipse = cv2.fitEllipse(contour)
            (xc,xy),(a,b),theta = ellipse
            # xc - x coord of center
            # yc - y coord of center
            # a - major semi-axis (half width)
            # b - minor semi-axis (half height)
            # theta - rotation angle
            #cv2.ellipse(img, ellipse, (0, 255, 0), 2)

            cv2.rectangle(img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,255,0,2))
            cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)

            cv2.putText(img, str(area)+': '+str(round(a,3))+', '+str(round(b,3)), (cent_x, cent_y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cropped_img = img[y-buffer:y+h+buffer, x-buffer:x+w+buffer]
            output_filename = output_dir+filename_split[0]+'_annotated_'+str(i)+'.'+filename_split[1]
            #cv2.imwrite(output_filename,cropped_img)
            #cv2.imshow(output_filename,cropped_img)
            i = i + 1

i = 0
for cran in myCrans:
    im_gray = cv2.cvtColor(cran, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow(str(i),cv2.bitwise_not(im_bw))
    i = i + 1

# displaying the image after drawing contours
cv2.imshow('Cranberries', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
