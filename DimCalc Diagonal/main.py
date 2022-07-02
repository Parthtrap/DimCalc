import math
import cv2
import IPH
import numpy as np


def Pythagorus(a, b):
    return math.sqrt((a[0]-b[0])**2 + ((a[1]-b[1])**2))


def MidPoint(a, b):
    return [int((a[0]+b[0])/2), int((a[1]+b[1])/2)]


# capture = cv2.VideoCapture(0)
# capture.set(10, 160)
# capture.set(3, 1920)
# capture.set(4, 1080)

# address = "http://192.168.211.248:8080/video"
# capture.open(address)

# while True:
#     success, img = capture.read()
#     img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
#     contours = IPH.getContours(img)

#     for contour in contours:
#         # contour = [len(approx), area, approx, bbox, i]

#         # Drawing Corner Points of Shape
#         for i in range(len(contour[2])):
#             cv2.circle(img, contour[2][i][0], 4, (255, 0, 0), -1)

#         # Drawing n-1 Edges of the shape
#         for i in range(len(contour[2])-1):
#             cv2.line(img, contour[2][i][0], contour[2]
#                      [i+1][0], (0, 0, 255), 2)
#             cv2.putText(img=img, text=str(int(Pythagorus(contour[2][i][0], contour[2][i+1][0]))),
#                         org=MidPoint(contour[2][i][0], contour[2][i+1][0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)

#         # Drawing the Last Edge
#         cv2.line(img, contour[2][len(contour[2])-1]
#                  [0], contour[2][0][0], (0, 0, 255), 2)
#         cv2.putText(img=img, text=str(int(Pythagorus(contour[2][len(contour[2])-1][0], contour[2][0][0]))),
#                     org=MidPoint(contour[2][len(contour[2])-1][0], contour[2][0][0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)

#     # DIsplaying both Original and Drawn image
#     cv2.imshow("Image", img)
#     cv2.waitKey(10)


img = cv2.imread("Test11.jpg")
img2 = img.copy()

contours = IPH.getContours(img)

for contour in contours:
    # cv2.drawContours(img2, contour[4], -1, (0, 0, 255), 4)
    # contour = [len(approx), area, approx, bbox, i]

    # Drawing Corner Points of Shape
    for i in range(len(contour[2])):
        cv2.circle(img2, contour[2][i][0], 4, (255, 0, 0), -1)

    # Drawing n-1 Edges of the shape
    for i in range(len(contour[2])-1):
        cv2.line(img2, contour[2][i][0], contour[2][i+1][0], (0, 0, 255), 2)
        cv2.putText(img=img2, text=str(int(Pythagorus(contour[2][i][0], contour[2][i+1][0]))),
                    org=MidPoint(contour[2][i][0], contour[2][i+1][0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)

    # Drawing the Last Edge
    cv2.line(img2, contour[2][len(contour[2])-1]
             [0], contour[2][0][0], (0, 0, 255), 2)
    cv2.putText(img=img2, text=str(int(Pythagorus(contour[2][len(contour[2])-1][0], contour[2][0][0]))),
                org=MidPoint(contour[2][len(contour[2])-1][0], contour[2][0][0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)

# Displaying both Original and Drawn image
IPH.DisplayMultiple({"Original": img, "Outline": img2})
