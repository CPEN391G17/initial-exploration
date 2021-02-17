import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
import os
import math
import keyboard


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def clean_LPNum(text, LP_type):
    if (LP_type == "IN"):
        num_groups = 4
    elif (LP_type == "NA"):
        num_groups = 6
    else:
        num_groups = 6

    regex = "[\s\d\w.:· -]+"
            #"[a-zA-Z0-9 .:-]+"

    text = text.replace("\n", "")
    result = re.search(regex, text)
    if (result == None):
        return ""
    if len(result.groups()) > num_groups:
        print("The num. groups is ", num_groups, " even after cleaning")

    return result.group()


def crop_plate(img, kernel, LP_type):
    plate_img = img.copy()
    if (LP_type == "IN"):
        classifier_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\indian_license_plate.xml"
    elif (LP_type == "NA"):
        classifier_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\haarcascade_russian_plate_number.xml"
    else:
        classifier_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\haarcascade_russian_plate_number.xml"

    # Cascade Classifier where our hundres of samples of license plates are
    plate_cascade = cv2.CascadeClassifier(classifier_addr)

    # gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.09, minNeighbors=11)
    #print(plate_rects)

    #return if no candidate found
    if len(plate_rects) == 0:
        return plate_img
    # draws the rectangle around it
    for (x, y, w, h) in plate_rects:
        x_offset = x
        y_offset = y

        x_end = x + w
        y_end = y + h

        # getting the points that show the license plate
        zoom_img = plate_img[y_offset:y_end, x_offset:x_end]
    return zoom_img

def select_plate(img, kernel, LP_type):
    plate_img = img.copy()
    if (LP_type == "IN"):
        classifier_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\indian_license_plate.xml"
    elif (LP_type == "NA"):
        classifier_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\haarcascade_russian_plate_number.xml"
    else:
        classifier_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\haarcascade_russian_plate_number.xml"

    # Cascade Classifier where our hundres of samples of license plates are
    plate_cascade = cv2.CascadeClassifier(classifier_addr)

    # gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=10)
    #print(plate_rects)

    # return if no candidate found
    if len(plate_rects) == 0:
        return plate_img
    # draws the rectangle around it
    (x, y, w, h) = plate_rects[0]
    x_offset = x
    y_offset = y

    x_end = x + w
    y_end = y + h

    # getting the points that show the license plate
    zoom_img = plate_img[y_offset:y_end, x_offset:x_end]
    # increasing the size of the image
    zoom_img = cv2.resize(zoom_img, (0, 0), fx=2, fy=2)
    zoom_img = zoom_img[7:-7, 7:-7]

    zy = (40 - (y_end - y_offset)) // 2
    zx = (144 - (x_end - x_offset)) // 2

    ydim = (y_end + zy - 50) - (y_offset - zy - 50)
    xdim = (x_end + zx) - (x_offset - zx)

    zoom_img = cv2.resize(zoom_img, (xdim, ydim))

    # putting the zoomed in image above where the license plate is located
    try:
        plate_img[y_offset - zy - 55:y_end + zy - 55, x_offset - zx:x_end + zx] = zoom_img
    except:
        pass

    # drawing a rectangle
    (x, y, w, h) = plate_rects[0]
    cv2.rectangle(plate_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return plate_img


def getYorN():
    while True:  # making a loop
        key = keyboard.read_key(suppress = True)
        if key == ("n") or key == ("y"):
            print("You pressed ", key)
            return keyboard.read_key()


def main(img_addr, LP_type):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img = None
    cropped = None
    text = None
    img = cv2.imread(img_addr, cv2.IMREAD_COLOR)

    scale_factor = math.sqrt((600*400)/(img.shape[0]*img.shape[1]))
    if (img.shape[0] > 400 or img.shape[1] > 600):
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #cv2.imshow('car img', img)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edge = cv2.Canny(gray, 30, 200)
    #cv2.imshow('edged', gray)
    #cv2.waitKey(0)

    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])

    cropped = crop_plate(img, kernel, LP_type)
    squared_plate = select_plate(img, kernel, LP_type)

    text = pytesseract.image_to_string(cropped, config='--psm 11')
    print("Raw text: ",text)

    text = clean_LPNum(text, LP_type)
    print("programming_fever's License Plate Recognition")
    print("Detected license plate Number is:", text, "\n")
    cv2.imshow(img_addr.replace(r"C:\Users\souno\GitRepos\CPEN391\LP-datasets",""), squared_plate)
    cv2.waitKey(0)

    return getYorN()
    '''
    #cv2.imshow('car', img)
    fig, axs = plt.subplots(2)
    fig.suptitle('Car and the cropped image')
    axs[0].imshow(img)
    axs[1].imshow(cropped)
    cv2.imshow('car', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

if __name__ == "__main__":
    # execute only if run as a script
    img_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\test1.jpeg"
    LP_type = "NA"

    mypath = r"C:\Users\souno\GitRepos\CPEN391\LP-datasets"
    allFiles = getListOfFiles(mypath)
    print(allFiles)

    dict = {}
    print(".jpg" in allFiles[1])
    for filepath in allFiles:
        if ".jpg" in filepath:
            print(filepath)
            if "NA-LPs" in filepath:
                bool = main(filepath, "NA")
                print(bool)
                dict[filepath] = bool
            '''
            elif "Indian-LPs" in filepath:
                bool = main(filepath, "IN")
                print(bool)
                dict[filepath] = bool
            '''

    test_name = input('Please enter the name of the test : ')
    file = open(test_name + ".txt", "w")
    for e in dict:
        L = [e.key, " : ", e.value]
        file.writelines(L)
    file.close()


    '''
    img_addr
    LP_type (choose one from the list) = {"IN", "NA"}
    '''
    #main(img_addr, LP_type)