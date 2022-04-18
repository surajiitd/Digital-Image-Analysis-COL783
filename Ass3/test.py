
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

# inp1_forest_SR_paper_euclidean_enhanced_p.png
# inp2_world_war2_SR_paper_euclidean_enhanced_p.png
# inp3_building_SR_paper_euclidean_enhanced_p.png
image_name = "inp3_building_SR_paper_euclidean_enhanced_p.png"
image = cv2.imread(image_name)
#cv2.imwrite("unsharp0.png",image)
# cv2.imshow("output",image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#METHOD-1  : unsharp_masking
ksize=(3,3)
sigma = 1.0
maskwt = 3.0 # weight of unsharp mask to be added to the image
image_caption = "maskwt={:.3f}".format(maskwt)
sharp_image = unsharp_masking(image,ksize=ksize, sigma=sigma, maskwt=maskwt)

#cv2.putText(sharp_image, image_caption, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#output_image_name = image_name+ #"unsharp_maskwt_{:.3f}.png".format(maskwt)
#cv2.imwrite(output_image_name  , sharp_image)
cv2.imwrite(image_name[:-4]+"_unsharp_masking.png",sharp_image)	
print("DONE with unsharp_masking")