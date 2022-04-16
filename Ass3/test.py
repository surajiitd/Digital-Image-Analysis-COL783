
# def hello():
# 	print("hello")
# 	calling("abc")

# def calling(st):
# 	print(st)
# hello()
import cv2
import numpy as np


def unsharp_masking(image,ksize=(3,3), sigma=1.0, maskwt=5.0):
	blurred_image = cv2.GaussianBlur(image, ksize, sigma)

	image = image.astype('int16')
	blurred_image = blurred_image.astype('int16')

	unsharp_mask = cv2.addWeighted(image, 1.0, blurred_image, -1.0, 0)
	cv2.imwrite("unsharp_mask.png" , unsharp_mask)

	#add the edge information to the image
	sharp_image = cv2.addWeighted(image, 1.0, unsharp_mask, maskwt, 0)
	#sharp_image = sharp_image-np.
	return sharp_image


image = cv2.imread("inp1_forest_SR_paper_euclidean.png")
cv2.imwrite("unsharp0.png",image)
# cv2.imshow("output",image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#METHOD-1  : unsharp_masking
ksize=(3,3)
sigma = 1.0
maskwt = 6.0 # weight of unsharp mask to be added to the image
image_caption = "maskwt={:.3f}".format(maskwt)
sharp_image_unsharp_masking = unsharp_masking(image,ksize=ksize, sigma=sigma, maskwt=maskwt)

cv2.putText(sharp_image_unsharp_masking, image_caption, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
output_image_name = "unsharp_maskwt_{:.3f}.png".format(maskwt)
cv2.imwrite(output_image_name  , sharp_image_unsharp_masking)
print("DONE with unsharp_masking")