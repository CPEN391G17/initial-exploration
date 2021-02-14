import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('img/image2.jpg') #read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale/recoloured img BGR->Gray
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) #converted to show since matplot expects a BGR image
plt.show()

#need to modify parameters
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction intensity
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

#contour detection to detect polygons within the edges/lines (in our case it is a rectangle)
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #return as a tree the approx version of the contours
contours = imutils.grab_contours(keypoints) #simplifies how contours are returned
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #sort the top 10 contour areas in descending order

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True) #how fine grain our contour is (10 modify?)
    if len(approx) == 4:
        location = approx
        break

print(location)

mask = np.zeros(gray.shape, np.uint8) #create a blank mask
new_image = cv2.drawContours(mask, [location], 0, 255, -1) #pass mask, contour
new_image = cv2.bitwise_and(img, img, mask=mask) #overlay mask over actual img
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()


(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
result

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()



