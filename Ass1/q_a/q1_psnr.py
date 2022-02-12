import cv2
import numpy as np
import math
import os 

def calculate_psnr(img, gt):
	(h,w,c) = img.shape
	diff = (gt-img) ** 2
	
	mse_b = np.sum(diff[:,:,0])/(h*w)
	mse_g = np.sum(diff[:,:,1])/(h*w)
	mse_r = np.sum(diff[:,:,2])/(h*w)

	print("mse = ",mse_b, mse_g, mse_r)

	psnr_b = round(10*math.log10(255**2 / mse_b), 4)
	psnr_g = round(10*math.log10(255**2 / mse_g), 4)
	psnr_r = round(10*math.log10(255**2 / mse_r), 4)
	print("PSNR = ", psnr_b, psnr_g, psnr_r)
	return (psnr_b, psnr_g, psnr_r)


if __name__=="__main__":
	dir_path = "data/demosaicing/1/"
	gt = cv2.imread( os.path.join(dir_path, "color.png") )
	matlab_img = cv2.imread(os.path.join(dir_path, "matlab/color.png"))
	cv2.imshow("matlab",matlab_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	matlab_img_psnr = calculate_psnr(matlab_img,gt)
	print("matlab PSNR",matlab_img_psnr)
	# print("Final image PSNR", final_img_psnr)
	# print("improvement over bilinear image = " ,[round(x-y,4) for x,y in zip(final_img_psnr, bil_img_psnr)])
