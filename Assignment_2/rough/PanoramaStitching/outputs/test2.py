import cv2
import numpy as np

# Read images : src image will be cloned into dst
im = cv2.imread("wood-texture.webp")
obj= cv2.imread("iloveyouticket.webp")
cv2.imwrite("wood-texture.jpg",im)
cv2.imwrite("iloveyouticket.jpg",obj)

im = cv2.imread("wood-texture.jpg")
obj= cv2.imread("iloveyouticket.jpg")
cv2.imshow("img",im)
cv2.imshow("obj",obj)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (int(height/2), int(width/2))

# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

# Write results
cv2.imwrite("opencv-normal-clone-example.jpg", normal_clone)
cv2.imwrite("opencv-mixed-clone-example.jpg", mixed_clone)