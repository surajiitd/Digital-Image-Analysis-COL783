
# def hello():
# 	print("hello")
# 	calling("abc")

# def calling(st):
# 	print(st)
# hello()
import cv2
import numpy as np

img = cv2.imread("inp3_building_SR_paper.png")
(h,w,c) = img.shape
# cv2.rectangle(img, (0,0), (w,h), (255,255,255),5)
img = img[5:h-5, 5:w-5]
img = cv2.resize(img,(w,h))
cv2.imshow("output",img)
cv2.waitKey()
cv2.destroyAllWindows()
