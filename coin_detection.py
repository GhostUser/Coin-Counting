import cv2 
import matplotlib.pyplot as plt 
import numpy as np

img=cv2.imread("coin_se.jpg")
#img=cv2.resize(img, (960, 540))  
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
ca=100
cb=200

ret, thres_tz= cv2.threshold(img_gray, 120, 255, cv2.THRESH_TOZERO)
blur_tz=cv2.medianBlur(thres_tz, 5, 0)
canny_tz=cv2.Canny(blur_tz, ca, cb)


canny_tz_1=cv2.threshold(canny_tz, 100, 255, cv2.THRESH_BINARY)[1]


test_img=canny_tz_1.copy()
test_img2=img.copy()


circles=cv2.HoughCircles(test_img,  cv2.HOUGH_GRADIENT, dp=2 ,minDist=250, minRadius=150, maxRadius=300)

circles=np.uint16(np.around(circles))

print(f"No. of Coins: {len(circles[0, :])}")

for i in circles[0, :]:
    test_img2=cv2.circle(test_img2, (i[0], i[1]), i[2], (255,0,0),9)
    test_img2=cv2.circle(test_img2, (i[0], i[1]), 10, (0,0,255), -1)
    print(f"Center: ({i[0]}, {i[1]}), Radius: {i[2]}")


cv2.imshow("Detected Coins", test_img2)
img_to_show=cv2.resize(test_img2, (1000, 750))
cv2.imshow("Detected Coins", img_to_show)
cv2.waitKey(0)
