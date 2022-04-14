import numpy as np
from const import *
from math import *
import cv2

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape=imgRGB.shape
    return np.dot(imgRGB.reshape(-1,3), yiq_from_rgb.transpose()).reshape(OrigShape)

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape=imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1,3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)

    pass

def build_pyramid(img):
	img_pyr = [None]*(NUMCELLS+1)  #start indexing from 1
	for level in range(1,MID):
		diff = level - MID
		
		new_size = (img.shape[1]*ALPHA**diff, img.shape[0]*ALPHA**diff)

		print("level - ",level)
		print("new size of image =",new_size)
		new_size = (floor(new_size[0] + EPSILON),floor(new_size[1] + EPSILON));
		print("new decim size of image = ",new_size)
		img_pyr[level] = cv2.resize(img,new_size,interpolation=cv2.INTER_LINEAR)
	img_pyr[MID] = img;
	for level in range(MID+1,NUMCELLS+1):
		diff = level - MID
		new_size = (img.shape[0]*ALPHA**diff, img.shape[1]*ALPHA**diff)
		print("level - ",level)
		print("new size of image =",new_size)
		new_size = (floor(new_size[0]),floor(new_size[1]));
		print("new decim size of image = ",new_size)
		img_pyr[level] = np.zeros(new_size)

	return img_pyr

def translate_img_by_half_pixel(img):
	(h,w) = img.shape
	img_new_y = np.zeros((h+1,w))
	img_new_y[1:,: ] = img
	img_new_y[0,:] = img_new_y[1,:]
	
	[X,Y] = np.meshgrid(list(range(0,w)),list(range(0,h)))
	X = X.astype(np.float32)
	Y = Y.astype(np.float32)
	Y = Y+.5
	out_y = cv2.remap(img_new_y, X,Y , cv2.INTER_CUBIC)

	img_new_x = np.zeros((h,w+1))
	img_new_x[:,1: ] = img
	img_new_x[:,0] = img_new_x[:,1]
	
	[X,Y] = np.meshgrid(list(range(0,w)),list(range(0,h)))
	X = X.astype(np.float32)
	Y = Y.astype(np.float32)
	X = X +.5
	out_x = cv2.remap(img_new_x, X,Y , cv2.INTER_CUBIC)
	
	return (out_x,out_y)

def img2patches(img):
	(hp,wp) = img.shape

	numPatches = hp*wp
	# patches_p = []
	# pi = []
	# pj = []
	patches_p = np.array([-1]*PATCH_SIZE,dtype=np.float32)
	pi = np.array([])
	pj = np.array([])
	pindex = 0
	for r in range(STEP,hp-STEP):
		for c in range(STEP, wp-STEP):
			pindex = pindex + 1
			p = img[r-STEP:r+STEP+1, c-STEP: c+STEP+1]
			#p = [pixel for row in p for pixel in row ]
			p = np.reshape(p,(-1))
			# patches_p.append(p)
			# pi.append(r)
			# pj.append(c)
			patches_p = np.vstack([patches_p,p])
			pi = np.append(pi,r)
			pj = np.append(pj,c)
	patches_p = np.delete(patches_p,0,0)
	return (patches_p,pi,pj)

def thresh( img, img_translated, patch_center_x,patch_center_y):
	p1 = coord2patch(img,patch_center_x,patch_center_y,STEP)
	p1 = coord2patch(img_translated,patch_center_x,patch_center_y,STEP)
	threshold = distance(p1,p2)
	return threshold

def coord2patch(img,i,j,step):
	psize = (step*2+1)**2
	(h,w) = img.shape
	p = []
	if (not checkbounds(h,w,i,j,step)):
		return
	p = img[i-step:i+step+1, j-step:j+step+1]
	p = np.reshape(p,(1,-1))

	return p

def distance(p1,p2):
	sigma = np.std(p2)
	if(sigma==0):
		sigma = MIN_STD
	ssd = np.sum((p1-p2)**2)
	dist = exp(-ssd/sigma)
	return dist

def checkbounds(h,w,i,j,step ):
	 if ( i-step < 0 or i+step > h-1 or j-step < 0 or j+step > w-1):
	 	is_center_of_any_patch = 0
	 else:
	 	is_center_of_any_patch = 1
	 return is_center_of_any_patch

def get_parent(imp,imq,qi,qj,factor):

	pj,pi = move_level(imq,qj,qi,imp)




def move_level(src_level,srcx,srcy,dst_level):
	#reverse to matlabs mapping function in imresize
	sh,sw = src_level.shape
	dh,dw = dst_level.shape
	scale_x = dw/sw
	scale_y = dh/sh

	dstx = scale_x*srcx - 0.5*scale_x*(1-1/scale_x)
	dsty = scale_y*srcy - 0.5*scale_y*(1-1/scale_y)

def set_parent():