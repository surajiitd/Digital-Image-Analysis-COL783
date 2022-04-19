import numpy as np
import cv2
import os
import math
import argparse

from utils.Airlight import Airlight
from utils.BoundCon import BoundCon
from utils.CalTransmission import CalTransmission
from utils.removeHaze import removeHaze


global image_id	

def unsharp_masking(image,ksize=(3,3), sigma=1.0, maskwt=5.0):
	blurred_image = cv2.GaussianBlur(image, ksize, sigma)

	image = image.astype('int16')
	blurred_image = blurred_image.astype('int16')

	unsharp_mask = cv2.addWeighted(image, 1.0, blurred_image, -1.0, 0)
	cv2.imwrite("{}/unsharp_mask.png".format(image_id) , unsharp_mask)

	#add the edge information to the image
	sharp_image = cv2.addWeighted(image, 1.0, unsharp_mask, maskwt, 0)
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

def dehaze(image, output_path, beta=0.425):
    HazeImg = image

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, beta)  #.85

    
    cv2.imwrite(output_path, HazeCorrectedImg)
    return HazeCorrectedImg

#commands to run
# python3 fog.py -i "../data/fog" -o "output"
if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inp', type=str, help='path to input directory containing dir 1,2,3')
	parser.add_argument('-o','--out', type=str, help='path to store output')
	args = parser.parse_args()

	global image_id
	image_id = int(input("enter the image id 1,2,3? : "))

	input_dir = os.path.join(args.inp, "{}".format(image_id) )
	image = cv2.imread(os.path.join(input_dir, "foggy.png"))
	output_dir = args.out   #"output"

	#write outputs
	print(os.path.join(output_dir,"{}".format(image_id)) )
	if not os.path.exists(os.path.join(output_dir,"{}/".format(image_id)) ):
		os.makedirs(os.path.join(output_dir,"{}/".format(image_id)))

	#METHOD-1  : unsharp_masking
	ksize=(3,3)
	sigma = 1.0
	maskwt = 5.0 # weight of unsharp mask to be added to the image
	image_caption = "maskwt={:.3f}".format(maskwt)
	sharp_image_unsharp_masking = unsharp_masking(image,ksize=ksize, sigma=sigma, maskwt=maskwt)

	cv2.putText(sharp_image_unsharp_masking, image_caption, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	output_image_name = "{}/1.unsharp_maskwt_{:.3f}.png".format(image_id, maskwt)
	cv2.imwrite(os.path.join(output_dir, output_image_name ) , sharp_image_unsharp_masking)
	print("DONE with METHOD-1  : unsharp_masking")



	#METHOD-2  : dehaze-> unsharp_masking

	# beta: parameter for dehazing
	if image_id==1:
		beta = .425
	elif image_id==2:
		beta = .5
	else:
		beta = .3

	maskwt = 4.0 if image_id==2 else 5.0
	dehaze_flag = True if input("do you have saved dehazed image y/n? : ")=='y' else False
	output_path = os.path.join(output_dir,"{}/dehaze_{:.3f}.jpg".format(image_id, beta) )

	if not dehaze_flag:
		dehazed_image = dehaze(image, output_path, beta=beta)
	else:
		dehazed_image = cv2.imread(output_path)
	
	image_caption = "dehaze_beta={:.3f}  maskwt={:.3f}".format(beta, maskwt)
	sharp_image_unsharp_masking = unsharp_masking(dehazed_image,ksize=ksize, sigma=sigma, maskwt=maskwt)

	cv2.putText(sharp_image_unsharp_masking, image_caption, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	output_image_name = "{}/2.dehaze_{:.3f}_unsharp_maskwt_{:.3f}.png".format(image_id, beta, maskwt) 
	cv2.imwrite(os.path.join(output_dir, output_image_name), sharp_image_unsharp_masking)
	print("DONE with METHOD-2  : dehaze -> unsharp_masking")
	



	#METHOD-3  : dehaze-> gamma_correction -> unsharp_masking
	gamma = 1.2 # used for gamma correction
	gamma_adjusted = perform_gamma_correction(dehazed_image, gamma)
	
	image_caption = "dehaze_beta={:.3f}  gamma={:.2f}  maskwt={:.3f}".format(beta, gamma, maskwt)
	sharp_image_unsharp_masking = unsharp_masking(gamma_adjusted,ksize=ksize, sigma=sigma, maskwt=maskwt)

	cv2.putText(sharp_image_unsharp_masking, image_caption, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	output_image_name = "{}/3.dehaze_{:.3f}_gamma={:.2f}_unsharp_maskwt_{:.3f}.png".format(image_id, beta, gamma, maskwt)
	cv2.imwrite(os.path.join(output_dir, output_image_name) , sharp_image_unsharp_masking)
	print("DONE with METHOD-3  : dehaze-> gamma_correction -> unsharp_masking")

	
	sharp_image_laplacian = sharp_using_Laplacian(image,ksize=3)
	cv2.putText(sharp_image_laplacian, "laplacian ksize={}".format(3), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	sharp_image_conv = sharp_using_Convolution(image)	
	cv2.putText(sharp_image_conv, "sharpening kernel".format(3), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

	
	cv2.imwrite(os.path.join(output_dir,"{}/image.png".format(image_id)) , image)
	cv2.imwrite(os.path.join(output_dir,"{}/sharp_image_laplacian.png".format(image_id) ), sharp_image_laplacian)
	cv2.imwrite(os.path.join(output_dir,"{}/sharp_image_conv.png".format(image_id)) , sharp_image_conv)
