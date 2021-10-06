import cv2 
import matplotlib.pyplot as plt 
import numpy as np

img=cv2.imread(sys.argv[1])
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth_gray=cv2.medianBlur(gray, 3)

cann=cv2.Canny(smooth_gray, 70, 140)

thres=cv2.threshold(cann, 130, 255, cv2.THRESH_TRUNC)[1]


circles=cv2.HoughCircles(thres,  cv2.HOUGH_GRADIENT, dp=2 ,minDist=425, minRadius=150, maxRadius=400)

circles=np.uint16(np.around(circles))

timg2=img.copy()

print(f"No. of cic: {len(circles[0, :])}")

for i in circles[0, :]:
    timg2=cv2.circle(timg2, (i[0], i[1]), i[2], (0,0,255), 9)
    timg2=cv2.circle(timg2, (i[0], i[1]), 10, (0,255,0), -1)
    print(f"Center: ({i[0]}, {i[1]}), Radius: {i[2]}")


cv2.imwrite(f"result_{sys.argv[-1]}", timg2)
