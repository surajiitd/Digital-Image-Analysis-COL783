import numpy as np
import cv2
import os
import math

global image_id	

def unsharp_masking(image,ksize=(3,3), sigma=2.0):
	blurred_image = cv2.GaussianBlur(image, (7, 7), 1.0)

	image = image.astype('int16')
	blurred_image = blurred_image.astype('int16')

	unsharp_mask = cv2.addWeighted(image, 1.0, blurred_image, -1.0, 0)
	cv2.imwrite("{}/unsharp_mask.png".format(image_id) , unsharp_mask)

	#add the edge information to the image
	sharp_image = cv2.addWeighted(image, 1.0, unsharp_mask, 4.0, 0)
	#sharp_image = sharp_image-np.
	return sharp_image

def sharp_using_Laplacian(image,ksize=3):
	ddepth = cv2.CV_16S
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	laplacian_kernel = np.array([[1, 1, 1],
                   				[1, -8,1],
                   				[1, 1, 1]])
	image = image.astype('int16')
	
	laplacian = cv2.filter2D(src=image, ddepth=-1, kernel=laplacian_kernel)

	cv2.imwrite("{}/laplacian.png".format(image_id) , laplacian)

	#add the edge information to the image
	sharp_image = cv2.addWeighted(image, 1.0, laplacian, -2, 0)
	return sharp_image

# Source: https://en.wikipedia.org/wiki/Kernel_(image_processing)
def sharp_using_Convolution(image): 
	sharpening_kernel = np.array([[0, -1, 0],
				                   [-1, 5,-1],
				                   [0, -1, 0]])

	sharp_image = cv2.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)
	return sharp_image


def generate_lookup_table(gamma):
    table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table

def perform_gamma_correction(image, gamma):
	table = generate_lookup_table(gamma)
	return cv2.LUT(image, table)



def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
def example():
    image = cv2.imread("result.jpg")
    sharpened_image = unsharp_mask(image)
    cv2.imwrite('my-sharpened-image.jpg', sharpened_image)



if __name__ == "__main__":

	image_set = "fog"
	global image_id
	image_id = int(input("enter the image id: "))

	dir_path = "../data/{}/{}".format(image_set, image_id)
	example()
	#image = cv2.imread(os.path.join(dir_path, "foggy.png"))
	image = cv2.imread( "result.jpg")
	ksize=(3,3)
	sigma = 1.0
	sharp_image_unsharp_masking = unsharp_masking(image,ksize=ksize, sigma=sigma)
	cv2.putText(sharp_image_unsharp_masking, "sigma={}_ksize={}".format(sigma, ksize), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

	cv2.imwrite("{}/sharp_image_unsharp_masking_{}_{}.png".format(image_id,sigma,ksize) , sharp_image_unsharp_masking)

	sharp_image_unsharp_masking = cv2.imread("{}/sharp_image_unsharp_masking_{}_{}.png".format(image_id,sigma,ksize))

	for gamma in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
		gamma_adjusted = perform_gamma_correction(sharp_image_unsharp_masking, gamma)
		cv2.putText(gamma_adjusted, "g={}".format(gamma), (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		cv2.imwrite("gamma_corrected_output_g_"+str(gamma)+".jpg", gamma_adjusted)

	sharp_image_laplacian = sharp_using_Laplacian(image,ksize=3)
	sharp_image_conv = sharp_using_Convolution(image)	
	print(type(sharp_image_unsharp_masking[0,0,0]))
	#write outputs
	if not os.path.exists("./{}".format(image_id) ):
		os.makedirs(str(image_id))

	sharpened_image = unsharp_mask(image)
	cv2.imwrite('{}/my-sharpened-image.png', sharpened_image)
	cv2.imwrite("{}/image.png".format(image_id) , image)
	cv2.imwrite("{}/sharp_image_unsharp_masking.png".format(image_id) , sharp_image_unsharp_masking)
	cv2.imwrite("{}/sharp_image_laplacian.png".format(image_id) , sharp_image_laplacian)
	cv2.imwrite("{}/sharp_image_conv.png".format(image_id) , sharp_image_conv)
