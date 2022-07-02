import math
import cv2

import IPH


img = cv2.imread("Test.jpg", cv2.IMREAD_REDUCED_COLOR_2)
print(IPH.CalcHist(img))
IPH.HistDisplay(img)


# greg "[t/fr]ed" ${filename}
# greg "[t/f]?ed" ${filename}
# greg "^[^g]" ${filename}
# greg "^(g|[0-9])" ${filename}
# greg "^(guna)" ${filename}
# greg "(sam).\1" ${filename}
