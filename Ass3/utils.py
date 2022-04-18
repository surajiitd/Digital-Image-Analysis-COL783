import numpy as np
from const import *
from math import *
import cv2
from scipy.spatial import distance
import scipy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


def transformRGB2YIQ(imRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    rows,cols,dims = imRGB.shape
    imYIQ = np.zeros((rows,cols,dims))


    imYIQ[:,:,0] = 0.299*imRGB[:,:,0] + 0.587 * imRGB[:,:,1] + 0.114 * imRGB[:,:,2];
    imYIQ[:,:,1] = 0.596 * imRGB[:,:,0] - 0.275 * imRGB[:,:,1] - 0.321 * imRGB[:,:,2];
    imYIQ[:,:,2] = 0.212 * imRGB[:,:,0] - 0.523 * imRGB[:,:,1] + 0.311 * imRGB[:,:,2];
   
    return imYIQ


def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transformMatrix = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    transformMatrix = np.linalg.inv(transformMatrix)
    (rows, cols, dims) = imYIQ.shape
    imRGB = np.zeros( (rows, cols, dims))
    
    imRGB[:, :, 0] = transformMatrix[0,0] * imYIQ[:,:,0] + transformMatrix[0,1] * imYIQ[:,:,1] + transformMatrix[0,2] * imYIQ[:,:,2]
    imRGB[:, :, 1] = transformMatrix[1,0] * imYIQ[:,:,0] + transformMatrix[1,1] * imYIQ[:,:,1] + transformMatrix[1,2] * imYIQ[:,:,2]
    imRGB[:, :, 2] = transformMatrix[2,0] * imYIQ[:,:,0] + transformMatrix[2,1] * imYIQ[:,:,1] + transformMatrix[2,2] * imYIQ[:,:,2]

    return imRGB

def build_pyramid(img):
	img_pyr = [None]*(NUMCELLS+1)  #start indexing from 1
	for level in range(1,MID):
		diff = level - MID
		
		new_size = (img.shape[1]*(ALPHA**diff), img.shape[0]*(ALPHA**diff))

		# print("level - ",level)
		# print("new size of image =",new_size)
		new_size = (floor(new_size[0] + EPSILON),floor(new_size[1] + EPSILON));
		# print("new decim size of image = ",new_size)
		img_pyr[level] = cv2.resize(img,new_size,interpolation=cv2.INTER_LINEAR)


	img_pyr[MID] = img;


	for level in range(MID+1,NUMCELLS+1):
		diff = level - MID
		new_size = (img.shape[0]*(ALPHA**diff), img.shape[1]*(ALPHA**diff))
		# print("level - ",level)
		# print("new size of image =",new_size)
		new_size = (floor(new_size[0]),floor(new_size[1]));
		# print("new decim size of image = ",new_size)
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
	pi = np.array([],dtype=np.uint8)
	pj = np.array([],dtype=np.uint8)
	pindex = 0
	for r in range(STEP,hp-STEP):
		for c in range(STEP, wp-STEP):
			pindex = pindex + 1
			p = img[r-STEP:r+STEP+1, c-STEP: c+STEP+1]
			p = np.reshape(p,(-1))

			patches_p = np.vstack([patches_p,p])
			pi = np.append(pi,r)
			pj = np.append(pj,c)
	patches_p = np.delete(patches_p,0,0)

	return (patches_p,pi,pj)

def thresh( img, img_translated, patch_center_x,patch_center_y):
	p1 = coord2patch(img,patch_center_x,patch_center_y,STEP)
	p2 = coord2patch(img_translated,patch_center_x,patch_center_y,STEP)
	threshold = distance(p1,p2)
	return threshold

def coord2patch(img,i,j,step):
	psize = (step*2+1)**2
	(h,w) = img.shape
	p = []
	if (not checkbounds(h,w,i,j,step)):
		return
	
	i = int(i)
	j = int(j)
	p = img[i-step:i+step+1, j-step:j+step+1]
	p = np.reshape(p,(-1))

	return p

def distance(p1,p2):
	sigma = np.std(p2)
	if(sigma==0):
		sigma = MIN_STD
	ssd =np.sum( np.sum((p1-p2)**2))
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

	(h,w) = imp.shape

	step = STEP
	if(checkbounds(h,w,pi,pj,step)):
		parent_patch = {'image':imp, 'pi': pi, 'pj':pj}
		b = True
	else:
		parent_patch = 0
		b = False
	return parent_patch,b



def move_level(src_level,srcx,srcy,dst_level):
	#reverse to matlabs mapping function in imresize
	sh,sw = src_level.shape
	dh,dw = dst_level.shape
	scale_x = dw/sw
	scale_y = dh/sh

	dstx = scale_x*srcx - 0.5*scale_x*(1-1/scale_x)
	dsty = scale_y*srcy - 0.5*scale_y*(1-1/scale_y)

	return dstx,dsty

def set_parent(curr_img, curr_pi, curr_pj, new_img,hr_example, factor,weighted_dists,sum_weights,lr_patch,lr_example):

	pj,pi = move_level(curr_img,curr_pj,curr_pi,new_img)

	(h,w) = new_img.shape

	step = STEP
	w_blur = W
	if(checkbounds(h,w,pi,pj,step)):
		#Rectangle to set
		left = ceil(pj-step)
		right = floor(pj+step)
		top = ceil(pi-step)
		bottom = floor(pi+step)

		Xqt = np.array(range(left,right+1))
		Yqt = np.array(range(top,bottom+1))
		dist_getx = Xqt - pj; 
		dist_gety = Yqt - pi;

		coord_x = hr_example['pj'] + dist_getx
		coord_y = hr_example['pi'] + dist_gety

		(X,Y) = np.meshgrid(coord_x,coord_y)

		X = X.astype(np.float32)
		Y = Y.astype(np.float32)
		patch = cv2.remap(hr_example['image'],X,Y, cv2.INTER_CUBIC)
		patch = np.clip(patch, a_min = 0, a_max = 1)

		weight = distance(lr_patch, lr_example)
		weights = weight
		sum_weights[top:bottom+1, left:right+1] = sum_weights[top:bottom+1, left:right+1] + weights
		weighted_dists[top:bottom+1, left:right+1] = weighted_dists[top:bottom+1, left:right+1] + patch * weights

	return (weighted_dists,sum_weights,new_img)

def unsharp_masking(image,ksize=(3,3), sigma=1.0, maskwt=5.0):
	blurred_image = cv2.GaussianBlur(image, ksize, sigma)

	image = image.astype('int16')
	blurred_image = blurred_image.astype('int16')
	unsharp_mask = cv2.addWeighted(image, 1.0, blurred_image, -1.0, 0)

	#add the edge information to the image
	sharp_image = cv2.addWeighted(image, 1.0, unsharp_mask, maskwt, 0)
	#sharp_image = sharp_image-np.
	return sharp_image

def euclidean_distance(x,y):
    return sqrt(np.sum((x-y)**2))


def cosine_distance(x,y):
    return 1-(np.dot(x,y)/(norm(x)*norm(y)))

def knnsearch_scikit(patches_db, input_patches,k, custom_distance_metric):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric = custom_distance_metric).fit(patches_db)
    distances, indices = nbrs.kneighbors(input_patches)
    return indices, distances

def knnsearch_scikit_brut(patches_db, input_patches,k, metric):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric = metric).fit(patches_db)
    distances, indices = nbrs.kneighbors(input_patches)
    return indices, distances

