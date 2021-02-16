import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
import os


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

    result = re.search("[a-zA-Z0-9.: ]+", text)
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

def main(img_addr, LP_type):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img = None
    cropped = None
    text = None
    img = cv2.imread(img_addr, cv2.IMREAD_COLOR)

    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])

    cropped = crop_plate(img, kernel, LP_type)

    text = pytesseract.image_to_string(cropped, config='--psm 11')
    print("Raw text: ",text)

    text = clean_LPNum(text, LP_type)
    print("programming_fever's License Plate Recognition")
    print("Detected license plate Number is:", text, "\n")


    #cv2.imshow('car', img)
    fig, axs = plt.subplots(2)
    fig.suptitle('Car and the cropped image')
    axs[0].imshow(img)
    axs[1].imshow(cropped)
    cv2.imshow('car', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # execute only if run as a script
    img_addr = r"C:\Users\souno\GitRepos\CPEN391\so_algo\test1.jpeg"
    LP_type = "NA"

    mypath = r"C:\Users\souno\GitRepos\CPEN391\LP-datasets"
    allFiles = getListOfFiles(mypath)
    print(allFiles)

    print(".jpg" in allFiles[1])
    for filepath in allFiles:
        if ".jpg" in filepath:
            print(filepath)
            if "NA-LPs" in filepath:
                main(filepath, "NA")
            elif "Indian-LPs" in filepath:
                main(filepath, "IN")
            cv2.waitKey(0)

    '''
    img_addr
    LP_type (choose one from the list) = {"IN", "NA"}
    '''
    #main(img_addr, LP_type)