import cv2
import numpy as np
import os
import time
import argparse
from q1_psnr import calculate_psnr

def find_color(i,j):
	if i%2==0 and j%2==0:
		return "b"
	elif i%2!=0 and j%2!=0:
		return "r"
	else:
		return "g"

def exists(i,j):
	(r,c) = img_shape
	if i<0 or i>=r or j<0 or j>=c : 
		return False
	else:
		return True


def interpolate_blue(color_img, bayer_img):
	for i in range(color_img.shape[0]):
		for j  in range(color_img.shape[1]):
			if find_color(i,j) == 'b':
				color_img[i,j,0] = bayer_img[i,j]
			elif (find_color(i,j)=='r' ):  #current pixel is red in bayer pattern
				sum = 0
				pixels = 0
				if( exists(i-1,j-1)):
					pixels+=1
					sum += bayer_img[i-1,j-1]
				if( exists(i-1,j+1)):
					pixels+=1
					sum += bayer_img[i-1,j+1]
				if( exists(i+1,j-1)):
					pixels+=1
					sum += bayer_img[i+1,j-1]
				if( exists(i+1,j+1)):
					pixels+=1
					sum += bayer_img[i+1,j+1]
				color_img[i,j,0] = int(round(sum/pixels))

			elif (i%2!=0 and j%2==0 ):
				sum = 0
				pixels = 0

				if( exists(i-1,j)):
					pixels+=1
					sum += bayer_img[i-1,j]

				if( exists(i+1,j)):
					pixels+=1
					sum += bayer_img[i+1,j]

				color_img[i,j,0] = int(round(sum/pixels))

			elif (i%2==0 and j%2!=0 ):
				sum = 0
				pixels = 0

				if( exists(i,j-1)):
					pixels+=1
					sum += bayer_img[i,j-1]

				if( exists(i,j+1)):
					pixels+=1
					sum += bayer_img[i,j+1]
					
				color_img[i,j,0] = int(round(sum/pixels))
		


def interpolate_green(color_img, bayer_img):
	for i in range(color_img.shape[0]):
		for j  in range(color_img.shape[1]):
			if (find_color(i,j) == 'g'):
				color_img[i,j,1] = bayer_img[i,j]
			else :
				sum = 0
				pixels = 0
				if( exists(i-1,j)):
					pixels+=1
					sum += bayer_img[i-1,j]
				if( exists(i+1,j)):
					pixels+=1
					sum += bayer_img[i+1,j]
				if( exists(i,j-1)):
					pixels+=1
					sum += bayer_img[i,j-1]
				if( exists(i,j+1)):
					pixels+=1
					sum += bayer_img[i,j+1]
				color_img[i,j,1] = int(round(sum/pixels))

def interpolate_red(color_img, bayer_img):
	for i in range(color_img.shape[0]):
		for j  in range(color_img.shape[1]):
			if find_color(i,j) == 'r':
				color_img[i,j,2] = bayer_img[i,j]
			elif (find_color(i,j) == 'b'):
				sum = 0
				pixels = 0
				if( exists(i-1,j-1)):
					pixels+=1
					sum += bayer_img[i-1,j-1]
				if( exists(i-1,j+1)):
					pixels+=1
					sum += bayer_img[i-1,j+1]
				if( exists(i+1,j-1)):
					pixels+=1
					sum += bayer_img[i+1,j-1]
				if( exists(i+1,j+1)):
					pixels+=1
					sum += bayer_img[i+1,j+1]
				color_img[i,j,2] = int(round(sum/pixels))
				
			elif (i%2==0 and j%2!=0 ):
				sum = 0
				pixels = 0

				if( exists(i-1,j)):
					pixels+=1
					sum += bayer_img[i-1,j]

				if( exists(i+1,j)):
					pixels+=1
					sum += bayer_img[i+1,j]

				color_img[i,j,2] = int(round(sum/pixels))

			elif (i%2!=0 and j%2==0 ):
				sum = 0
				pixels = 0

				if( exists(i,j-1)):
					pixels+=1
					sum += bayer_img[i,j-1]

				if( exists(i,j+1)):
					pixels+=1
					sum += bayer_img[i,j+1]
					
				color_img[i,j,2] = int(round(sum/pixels))
			



def bilinear_interpolation(color_img, bayer_img):
	interpolate_blue(color_img, bayer_img)
	interpolate_green(color_img, bayer_img)
	interpolate_red(color_img, bayer_img)

g_factor = {
	'r': 1/2,
	'g': 5/8,
	'b': 3/4
}

def add_delta_in_blue_channel(color_img, bayer_img):
	for i in range(color_img.shape[0]):
		for j in range(color_img.shape[1]):
			curr_color = find_color(i,j)
			actual_value = bayer_img[i,j]

			#calculate interpolated value at current pixel
			if( curr_color == 'g'):
				sum = 0
				pixels = 0
				#pixels who always are of 1 weight
				if( exists(i-1,j-1)):
					pixels+=1
					sum += bayer_img[i-1,j-1]
				if( exists(i-1,j+1)):
					pixels+=1
					sum += bayer_img[i-1,j+1]
				if( exists(i+1,j-1)):
					pixels+=1
					sum += bayer_img[i+1,j-1]
				if( exists(i+1,j+1)):
					pixels+=1
					sum += bayer_img[i+1,j+1]

				#pixels with some half weights
				half_wt = 'hor' if (i%2!=0 and j%2==0) else 'vert'

				if( exists(i-2,j)):
					pixels+= 0.5 if half_wt=='vert' else 1
					sum += bayer_img[i-2,j] * (0.5 if half_wt=='vert' else 1)
				if( exists(i+2,j)):
					pixels+=0.5 if half_wt=='vert' else 1
					sum += bayer_img[i+2,j] * (0.5 if half_wt=='vert' else 1)
				if( exists(i,j-2)):
					pixels+=0.5 if half_wt=='hor' else 1
					sum += bayer_img[i,j-2] * (0.5 if half_wt=='hor' else 1)
				if( exists(i,j+2)):
					pixels+=0.5 if half_wt=='hor' else 1
					sum += bayer_img[i,j+2] * (0.5 if half_wt=='hor' else 1)

				interpolated_value = sum/pixels

				# add the delta term in the bilinear interpolated value
				delta_term =  round(g_factor[curr_color] * (actual_value - interpolated_value))
				if (color_img[i,j,0] + delta_term > 255 ):
					color_img[i,j,0] = 255
				elif(color_img[i,j,0] + delta_term < 0):
					color_img[i,j,0] = 0
				else:
					color_img[i,j,0] += delta_term

			if( curr_color == 'r'):
				sum = 0
				pixels = 0
				
				if( exists(i-2,j)):
					pixels+=1
					sum += bayer_img[i-2,j]
				if( exists(i+2,j)):
					pixels+=1
					sum += bayer_img[i+2,j]
				if( exists(i,j-2)):
					pixels+=1
					sum += bayer_img[i,j-2]
				if( exists(i,j+2)):
					pixels+=1
					sum += bayer_img[i,j+2]

				

				interpolated_value = sum/pixels
				
				# add the delta term in the bilinear interpolated value
				delta_term =  round(g_factor[curr_color] * (actual_value - interpolated_value))
				if (color_img[i,j,0] + delta_term > 255 ):
					color_img[i,j,0] = 255
				elif(color_img[i,j,0] + delta_term < 0):
					color_img[i,j,0] = 0
				else:
					color_img[i,j,0] += delta_term
	

def add_delta_in_green_channel(color_img, bayer_img):
	for i in range(color_img.shape[0]):
		for j in range(color_img.shape[1]):
			curr_color = find_color(i,j)
			if( curr_color != 'g'):
				actual_value = bayer_img[i,j]

				##calculate interpolated value at current pixel
				sum = 0
				pixels = 0
				if( exists(i-2,j)):
					pixels+=1
					sum += bayer_img[i-2,j]
				if( exists(i+2,j)):
					pixels+=1
					sum += bayer_img[i+2,j]
				if( exists(i,j-2)):
					pixels+=1
					sum += bayer_img[i,j-2]
				if( exists(i,j+2)):
					pixels+=1
					sum += bayer_img[i,j+2]

				interpolated_value = sum/pixels
				
				# add the delta term in the bilinear interpolated value
				delta_term =  round(g_factor[curr_color] * (actual_value - interpolated_value))
				if (color_img[i,j,1] + delta_term > 255 ):
					color_img[i,j,1] = 255
				elif(color_img[i,j,1] + delta_term < 0):
					color_img[i,j,1] = 0
				else:
					color_img[i,j,1] += delta_term

			

def add_delta_in_red_channel(color_img, bayer_img):
	for i in range(color_img.shape[0]):
		for j in range(color_img.shape[1]):
			curr_color = find_color(i,j)
			actual_value = bayer_img[i,j]

			#calculate interpolated value at current pixel
			if( curr_color == 'g'):
				sum = 0
				pixels = 0
				#pixels who always are of 1 weight
				if( exists(i-1,j-1)):
					pixels+=1
					sum += bayer_img[i-1,j-1]
				if( exists(i-1,j+1)):
					pixels+=1
					sum += bayer_img[i-1,j+1]
				if( exists(i+1,j-1)):
					pixels+=1
					sum += bayer_img[i+1,j-1]
				if( exists(i+1,j+1)):
					pixels+=1
					sum += bayer_img[i+1,j+1]

				#pixels with some half weights
				half_wt = 'hor' if (i%2==0 and j%2!=0) else 'vert'

				if( exists(i-2,j)):
					pixels+= 0.5 if half_wt=='vert' else 1
					sum += bayer_img[i-2,j] * (0.5 if half_wt=='vert' else 1)
				if( exists(i+2,j)):
					pixels+=0.5 if half_wt=='vert' else 1
					sum += bayer_img[i+2,j] * (0.5 if half_wt=='vert' else 1)
				if( exists(i,j-2)):
					pixels+=0.5 if half_wt=='hor' else 1
					sum += bayer_img[i,j-2] * (0.5 if half_wt=='hor' else 1)
				if( exists(i,j+2)):
					pixels+=0.5 if half_wt=='hor' else 1
					sum += bayer_img[i,j+2] * (0.5 if half_wt=='hor' else 1)

				interpolated_value = sum/pixels

				# add the delta term in the bilinear interpolated value
				delta_term =  round(g_factor[curr_color] * (actual_value - interpolated_value))
				if (color_img[i,j,2] + delta_term > 255 ):
					color_img[i,j,2] = 255
				elif(color_img[i,j,2] + delta_term < 0):
					color_img[i,j,2] = 0
				else:
					color_img[i,j,2] += delta_term

			if( curr_color == 'b'):
				sum = 0.
				pixels = 0.
				
				if( exists(i-2,j)):
					pixels+=1
					sum += bayer_img[i-2,j]
				if( exists(i+2,j)):
					pixels+=1
					sum += bayer_img[i+2,j]
				if( exists(i,j-2)):
					pixels+=1
					sum += bayer_img[i,j-2]
				if( exists(i,j+2)):
					pixels+=1
					sum += bayer_img[i,j+2]

				

				interpolated_value = sum/pixels
				
				# add the delta term in the bilinear interpolated value
				delta_term =  round(g_factor[curr_color] * (actual_value - interpolated_value))
				if (color_img[i,j,2] + delta_term > 255 ):
					color_img[i,j,2] = 255
				elif(color_img[i,j,2] + delta_term < 0):
					color_img[i,j,2] = 0
				else:
					color_img[i,j,2] += delta_term


def add_delta_term(color_img, bayer_img):
	add_delta_in_blue_channel(color_img, bayer_img)
	add_delta_in_green_channel(color_img, bayer_img)
	add_delta_in_red_channel(color_img, bayer_img)
	
	
# Command : python3 q1.py -i "../data/demosaicing"
if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inp', type=str, help='path to input directory containing dir 1,2,3,4', default="../data/demosaicing")
	args = parser.parse_args()

	img_num = input("which image 1,2,3,4 ?: ")
	dir_path = os.path.join(args.inp, "{}/".format(img_num) )

	bayer_img = cv2.imread( os.path.join(dir_path, "bayer.jpg"),cv2.IMREAD_GRAYSCALE)
	
	global img_shape
	img_shape = bayer_img.shape

	color_img = np.zeros(bayer_img.shape + (3,), dtype='uint8')

	#checking to print the bayer pattern
	print("our Bayer Pattern:")
	for i in range(4):
		for j in range(4):
			print(find_color(i,j),end=' ')
		print()
	

	
	#Mosaiced image – just coded with the Bayer filter colors
	bayer_coded_color_img = cv2.merge([bayer_img, bayer_img, bayer_img])
	
	for i in range(bayer_img.shape[0]):
		for j in range(bayer_img.shape[1]):
			if(find_color(i,j)!='b'):
				bayer_coded_color_img[i,j,0] = 0
			if(find_color(i,j)!='g'):
				bayer_coded_color_img[i,j,1] = 0
			if(find_color(i,j)!='r'):
				bayer_coded_color_img[i,j,2] = 0

	cv2.imshow("bayer image",bayer_img)
	cv2.imshow("No interp",bayer_coded_color_img)
	cv2.waitKey(4000)
	cv2.destroyAllWindows()
	cv2.imwrite(os.path.join(dir_path, "No_interp.png") ,bayer_coded_color_img)


	#bilinear interpolation on image
	print("Demosaicing...")
	start_time = time.time()
	bilinear_interpolation(color_img, bayer_img)
	bil_img = np.copy(color_img)

	#add delta term on bilinear interpolated image
	add_delta_term(color_img, bayer_img)
	print("Time taken: {:.2f} sec".format(time.time()-start_time) )

	gt = cv2.imread( os.path.join(dir_path, "color.png") )
	matlab_img = cv2.imread(os.path.join(dir_path, "matlab.png"))
	cv2.imshow("only bilinear interp",bil_img)
	cv2.imshow("groundtruth image",gt)
	cv2.imshow("matlab_img",matlab_img)
	cv2.imshow("Final color image",color_img)
	cv2.imwrite(os.path.join(dir_path, "only_bilinear_interp.png") ,bil_img)
	cv2.imwrite(os.path.join(dir_path, "bilinear_interp_with_delta.png") ,color_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	final_img = cv2.imread(os.path.join(dir_path, "bilinear_interp_with_delta.png"))

	bil_img_psnr = calculate_psnr(bil_img,gt)
	final_img_psnr = calculate_psnr(final_img,gt)
	print("only bilinear interp PSNR",bil_img_psnr)
	print("Final image PSNR", final_img_psnr)
	print("improvement over bil image = " ,[round(x-y,4) for x,y in zip(final_img_psnr, bil_img_psnr)])

	print("\nMATLAB:")
	matlab_img_psnr = calculate_psnr(matlab_img,gt)
	print("matlab PSNR:",matlab_img_psnr)
	print("improvement over bil image = " ,[round(x-y,4) for x,y in zip(matlab_img_psnr, bil_img_psnr)])


