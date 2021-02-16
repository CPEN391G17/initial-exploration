import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re

def correct_LP(text):
    result = re.search(r"[a-zA-z0-9]+", text)
    if len(result.group(0)) > 7:
        print("The length is not 7 even after cleaning")

    return result.group(0)


def crop_plate(img,kernel):
    plate_img = img.copy()

    # gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=10)
    print(plate_rects)

    # draws the rectangle around it
    for (x, y, w, h) in plate_rects:
        x_offset = x
        y_offset = y

        x_end = x + w
        y_end = y + h

        # getting the points that show the license plate
        zoom_img = plate_img[y_offset:y_end, x_offset:x_end]
    return zoom_img

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread(r"C:\Users\souno\GitRepos\CPEN391\so_algo\test1.jpeg", cv2.IMREAD_COLOR)
#img = cv2.resize(img, (600, 400))
#cv2.imshow('car', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)


edged = cv2.Canny(gray, 30, 200)
#cv2.imshow('edge detection', edged)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

#print(contours)
for c in contours:

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 5, True)
    if len(approx) == 4:
        screenCnt = approx
        break

'''
if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)


mask = np.zeros(gray.shape, np.uint8) #create a blank mask
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1) #pass mask, contour
new_image = cv2.bitwise_and(img, img, mask=mask) #overlay mask over actual img
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

'''

# Cascade Classifier where our hundres of samples of license plates are
plate_cascade = cv2.CascadeClassifier(
    #r"C:\Users\souno\GitRepos\CPEN391\so_algo\indian_license_plate.xml"
    r"C:\Users\souno\GitRepos\CPEN391\so_algo\haarcascade_russian_plate_number.xml"
)


kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])

Cropped = crop_plate(img, kernel)
'''
# Cropping
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
plt.imshow(cv2.cvtColor(Cropped, cv2.COLOR_BGR2RGB))
plt.show()
'''

text = pytesseract.image_to_string(Cropped, config='--psm 11')
text = correct_LP(text)
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is:", text)

img = cv2.resize(img, (500, 300))
Cropped = cv2.resize(Cropped, (400, 200))
cv2.imshow('car', img)
cv2.imshow('Cropped', Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()