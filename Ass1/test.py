import numpy as np
import cv2
import os
import math
from q_a.q1_psnr import calculate_psnr

red = np.zeros((500, 500, 3))
red[:,:,0] = 255
cyan = np.zeros((500, 500, 3)) + 255 - red
cv2.imshow("red",red)
cv2.imshow("cyan",cyan)
cv2.waitKey()
cv2.destroyAllWindows()

##########################################  TA
# while(True):
# 	ch = input("continue : y/n")
# 	if(ch=='n'):
# 		break
# 	n = int(input("enter no. of int nodes:"))
# 	n = n+(n+1)
# 	h = math.log2(n+1)-1
# 	print("height: ",h)

# arr = [10, 6, 0, 7, 5, 8, 1, 15, 2, 8, 11, 12]
# arr2 = [10, 4, 0, 7, 5, 8, 1, 13, 2, 6, 11, 10]
# arr3 = [12, 4, 0, 7, 5, 8, 1, 13, 4, 6, 13, 10]



# arr4 = [12, 4, 0, 9, 5, 8, 1, 13, 6, 8, 11, 10]

# sum = 0
# for i,x in enumerate(arr4):
# 	sum += (i+1)*x
# print(sum)



