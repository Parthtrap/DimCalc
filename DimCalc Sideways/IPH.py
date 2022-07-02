import cv2
from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt


def Display(title, src):
    cv2.imshow(title, src)
    cv2.waitKey(0)


def Grayscale(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


def ListDisplay(titleList):
    for i in range(len(titleList)):
        cv2.imshow(str(i), titleList[i])
    cv2.waitKey(0)


def DisplayMultiple(srcDict):
    for i in srcDict:
        cv2.imshow(i, srcDict[i])
    cv2.waitKey(0)


def Resize(src, x, y, InterPolation=None):
    if(InterPolation != None):
        return cv2.resize(src, (x, y), interpolation=InterPolation)
    else:
        return cv2.resize(src, (x, y))


def Scale(src, x, y):
    return cv2.resize(src, (0, 0), fx=x, fy=y)


def Outline(src, x=None, y=None):
    if(x != None and y != None):
        return cv2.Canny(src, x, y)
    else:
        return cv2.Canny(src, 100, 100)


def Blur(src, n, y=None, z=None):
    if(y != None and z != None):
        return cv2.bilateralFilter(src, n, y, z)
    if(y != None and z == None):
        return cv2.GaussianBlur(src, (n, y), 1)
    else:
        return cv2.medianBlur(src, n)


def Erode(src, x, y, Iterations=1):
    return cv2.erode(src, np.ones((x, y), np.uint8), iterations=Iterations)


def Dilate(src, x, y, Iterations=1):
    return cv2.dilate(src, np.ones((x, y), np.uint8), iterations=Iterations)


def CalcHist(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def HistDisplay(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", img_gray)
    plt.plot(CalcHist(img))
    plt.show()
    cv2.waitKey(0)


def BinaryThresh(src, x=128):
    return cv2.threshold(Grayscale(src), x, 255, cv2.THRESH_BINARY)[1]


def InvBinaryThresh(src, x=128):
    return cv2.threshold(Grayscale(src), x, 255, cv2.THRESH_BINARY_INV)[1]


def TruncThresh(src, x=128):
    return cv2.threshold(Grayscale(src), x, 255, cv2.THRESH_TRUNC)[1]


def ToZeroThresh(src, x=128):
    return cv2.threshold(Grayscale(src), x, 255, cv2.THRESH_TOZERO)[1]


def InvToZeroThresh(src, x=128):
    return cv2.threshold(Grayscale(src), x, 255, cv2.THRESH_TOZERO_INV)[1]


def AdaptiveThresh(src, x=128):
    return cv2.adaptiveThreshold(Grayscale(src), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)


def OtsuThresh(src, x=128):
    return cv2.threshold(Grayscale(src), x, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def colorFilterMask(src, lower, upper):
    return cv2.inRange(cv2.cvtColor(src, cv2.COLOR_BGR2HSV), lower, upper)


def colorFilter(src, lower, upper):
    return cv2.bitwise_and(src, src, mask=colorFilterMask(src, lower, upper))


def HoughLines(src):
    img1 = src.copy()
    lines = (cv2.HoughLines(cv2.Canny(Grayscale(src), 50,
             150, apertureSize=3), 1, np.pi/180, 200))
    if lines is None:
        res = "Not Found"
    else:
        for r_theta in lines[0]:
            r, theta = r_theta
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        res = "Found"
    return res, img1


def ContourFind(img):
    return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]


def getContours(img, cannyThresh=[30, 30], min_area=100, sides=0):
    img_canny = Outline(Blur(Grayscale(img), 3, 3),
                        cannyThresh[0], cannyThresh[1])
    img_dilate = Dilate(img_canny, 4, 4, 1)
    img_eroded = Erode(img_dilate, 4, 4, 1)
    imshow("Canny", img_eroded)
    final_contours = []
    contours = ContourFind(img_eroded)
    for i in contours:
        area = cv2.contourArea(i)
        if area > min_area:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.005*peri, True)
            bbox = cv2.boundingRect(approx)
            if sides > 0:
                if(len(approx) == sides):
                    final_contours.append([len(approx), area, approx, bbox, i])
            else:
                final_contours.append([len(approx), area, approx, bbox, i])

    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)
    return final_contours
