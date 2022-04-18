import cv2
import os
from utils import *
from const import *
import pyflann
import numpy as np


# Choices for distance metric are: 'euclidean'(or l2), 'manhattan'(or l1), 'cosine'(for cosine distances),  
distance_metric = 'euclidean'  #'euclidean', 'manhattan', 'cosine'


def SR(img):

	img_pyr = build_pyramid(img)
	highest_lvl_filled = MID;
	out_x, out_y = translate_img_by_half_pixel(img)

	#Init patch database, according to current highest filled layer.
	patches_db = np.array([-1]*PATCH_SIZE,dtype=np.float32)
	qi = np.array([])
	qj = np.array([])
	qlvl = np.array([])
	print("Building patch database...")
	for lvlq in range(MID-1,0,-1):
		(input_patches_q,lvlq_pi, lvlq_pj) = img2patches(img_pyr[lvlq])
		patches_db = np.vstack([patches_db, input_patches_q])
		qi = np.append(qi,lvlq_pi)
		qj = np.append(qj,lvlq_pj)

		levels = np.ones(( len(lvlq_pi) ),dtype=np.uint8)  *  lvlq
		qlvl = np.append(qlvl, levels)
	patches_db = np.delete(patches_db,0,0)

	(input_patches_p,input_pi,input_pj) = img2patches(img_pyr[MID])

	print("LENGTH of 1 PATCH = ",len(input_patches_p[0]))

	next_target_start = MID+1;
	tot_numPatches = len(patches_db)
	query_numPatches = len(input_patches_p)
	print("Total no. of patches = ",tot_numPatches)
	print("no. of query patches = ",query_numPatches)
	
	print("Performing KNN Search...")
	print("For K = {}".format(K))

	#Find k nearest neighbor patch of pathches in current image in the image pyramid (from mid-1 to lowest level)
	if(distance_metric=='euclidean' or distance_metric=='manhattan'):
		NNs, Dist = knnsearch_scikit(patches_db, input_patches_p,k=K, custom_distance_metric=distance_metric)
	elif (distance_metric =='cosine' or distance_metric=='correlation'):  
		NNs, Dist = knnsearch_scikit_brut(patches_db, input_patches_p,k=K, metric=distance_metric)


	for next_target in range(next_target_start,NUMCELLS+1):
		skipped = 0
		skipped_no_info=0
		delta = next_target-MID
		(htg,wtg) = img_pyr[next_target].shape	

		new_img = np.ones((htg,wtg))*DEFAULT_BG_GREYVAL
		weighted_dists = np.zeros((htg,wtg))
		sum_weights = np.zeros((htg,wtg))
		factor_src = htg/img.shape[0]

		no_of_query_patches = len(NNs)
		for p_idx in range(0,no_of_query_patches):
			if(p_idx % 1000 == 0):
				print("progress: patch {}/{}".format(p_idx,no_of_query_patches))


			pknns = NNs[p_idx]
			# check if the patch nn passes the min thresh criteria
			t_x = thresh( img, out_x, input_pi[p_idx],input_pj[p_idx] )
			t_y = thresh( img, out_y, input_pi[p_idx],input_pj[p_idx] )
			t = (t_x+t_y)/2

			#taken is counts the amount of predictions for current lr example which were set
			taken  = 0
			for k in range(0,K):

				nn = pknns[k]

				nnParentlvl = qlvl[nn] + delta
				if(nnParentlvl > highest_lvl_filled):
					skipped_no_info += 1
					continue


				#parent patch
				imp = img_pyr[int(qlvl[nn] + delta)]
				parent_h = imp.shape[0]

				#matched nearest neighbour
				imq = img_pyr[int(qlvl[nn])]
				child_h = imq.shape[0]

				factor_example = parent_h/child_h

				if(taken > 0 and Dist[p_idx,k] >= t):
					skipped += 1
					continue

				lr_patch = input_patches_p[p_idx,:]
				lr_example = patches_db[nn,:]
				lr_example = np.reshape(lr_example,(-1,W))

				(hr_example,b) = get_parent(imp,imq,qi[nn],qj[nn],factor_example)

				if(b):
					taken += 1
					(weighted_dists, sum_weights, new_img) = \
						set_parent(img,input_pi[p_idx], input_pj[p_idx], \
						new_img,hr_example, factor_src, weighted_dists, \
						 sum_weights, lr_patch, patches_db[nn,:] ) 

		new_img = weighted_dists/sum_weights
		new_img[ np.isnan(new_img) ] = 0

		img_pyr[next_target] = new_img
		highest_lvl_filled = next_target

	output = new_img
	return output




if __name__ == "__main__":

	images = ["inp1_forest.png","inp2_world_war2.png", "inp3_building.png"]

	#output_resolutions = [ (552, 296), (610, 385), (487,261)]
	#output_resolutions = [ (296,552), (385,610), (261,487)]

	for i in range(len(images)):
	
		img = cv2.imread(os.path.join("Assignment3_data",images[i]) )
		img2 = img
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		output_resolution = (img.shape[1]*SCALE,img.shape[0]*SCALE)

		#Nearest Neighbor Interpolation Method
		img_nearest = cv2.resize(img2,output_resolution, interpolation=cv2.INTER_NEAREST)
		cv2.imwrite(images[i][:-4]+"_nearest.png", img_nearest)

		#BiCubic Interpolation Method
		img_bicubic = cv2.resize(img2,output_resolution, interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(images[i][:-4]+"_bicubic.png", img_bicubic)


		img_bicubic = cv2.resize(img,output_resolution, interpolation=cv2.INTER_CUBIC)
		yiq_img_big = transformRGB2YIQ(img_bicubic)

		yiq_orig_img = transformRGB2YIQ(img)

		grey_img = yiq_orig_img[:,:,0] 
		grey_img = grey_img.astype(np.float32)
		grey_img = grey_img/255

		grey_SRed = SR(grey_img)


		grey_SRed *= 255
		yiq_img_big[:,:,0] = grey_SRed
		img_SRed = transformYIQ2RGB(yiq_img_big)
		img_SRed = img_SRed.astype(np.uint8)
		img_SRed = cv2.cvtColor(img_SRed, cv2.COLOR_RGB2BGR)
		(h,w,_) = img_SRed.shape
		img_SRed = img_SRed[5:h-5, 5:w-5]
		img_SRed = cv2.resize(img_SRed,(w,h))

		cv2.imwrite(images[i][:-4]+"_SR_paper_{}.png".format(distance_metric),img_SRed)


		## DO UNSHARP MASKING
		ksize=(3,3)
		sigma = 1.0
		maskwt = 3.0 # weight of unsharp mask to be added to the image
		image_caption = "maskwt={:.3f}".format(maskwt)
		sharp_image = unsharp_masking(img_SRed,ksize=ksize, sigma=sigma, maskwt=maskwt)
		cv2.imwrite(images[i][:-4]+"_SR_paper_{}_unsharp_masking.png".format(distance_metric),sharp_image)		

		cv2.putText(sharp_image, image_caption, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		print("DONE with unsharp_masking")
		cv2.imwrite(images[i][:-4]+"_SR_paper_{}_unsharp_masking_{}.png".format(distance_metric,image_caption),sharp_image)		
		
