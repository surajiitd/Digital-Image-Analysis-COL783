import cv2
import os
from utils import *
from const import *
import argparse
#from scipy.spatial import distance

def SR(img,distance_metric):

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

	print("input_pi = ",input_pi)
	print("LENGTH of 1 PATCH = ",len(input_patches_p[0]))

	next_target_start = MID+1;
	tot_numPatches = len(patches_db)
	query_numPatches = len(input_patches_p)
	print("Total no. of patches = ",tot_numPatches)
	print("no. of query patches = ",query_numPatches)
	
	print("Performing KNN Search...")
	print("For K = {}".format(K))


	NNs, Dist = knnsearch_scikit(patches_db, input_patches_p,k=K, custom_distance_metric=distance_metric)
	#Dist = np.sqrt(Dist)


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


def main_SR(img,distance_metric):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	output_resolution = (img.shape[1]*SCALE,img.shape[0]*SCALE)

	img_bicubic = cv2.resize(img,output_resolution, interpolation=cv2.INTER_CUBIC)
	yiq_img_big = transformRGB2YIQ(img_bicubic)
	yiq_orig_img = transformRGB2YIQ(img)

	grey_img = yiq_orig_img[:,:,0] 

	grey_img = grey_img/255
	grey_SRed = SR(grey_img,distance_metric)
	grey_SRed *= 255
	yiq_img_big[:,:,0] = grey_SRed

	img_SRed = transformYIQ2RGB(yiq_img_big)
	img_SRed = img_SRed.astype(np.uint8)
	img_SRed = cv2.cvtColor(img_SRed, cv2.COLOR_RGB2BGR)
	(h,w,_) = img_SRed.shape
	img_SRed = img_SRed[5:h-5, 5:w-5]
	img_SRed = cv2.resize(img_SRed,(w,h))
	return img_SRed


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inp', type=str, help='path to input image')
	parser.add_argument('-o','--out', type=str, help='path to store output images')
	parser.add_argument('-d','--dist', type=str, default='euclidean', help='distance_metric among : euclidean, manhattan, cosine, correlation')
	args = parser.parse_args()

	
	input_path = args.inp
	output_path = args.out
	distance_metric = args.dist

	img = cv2.imread( input_path )
	
	# Apply different transformations to Low Resolution image.
	orig = img
	rotated90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	rotated180 = cv2.rotate(img, cv2.ROTATE_180)
	rotated270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	flipped = cv2.flip(img,0)   # 0 represents flipping vertically (upside down)
	rotated90_flipped = cv2.flip(rotated90,0)  
	rotated180_flipped = cv2.flip(rotated180,0)  
	rotated270_flipped = cv2.flip(rotated270,0)  
	
	
	# Call the SR function for all transformed images.
	orig = main_SR(orig,distance_metric)
	cv2.imwrite(os.path.join(output_path,"{}.png".format(distance_metric)),orig)
	rotated90 = main_SR(rotated90,distance_metric)
	cv2.imwrite(os.path.join(output_path,"rotated90_{}.png".format(distance_metric)),rotated90)
	rotated180 = main_SR(rotated180,distance_metric)
	cv2.imwrite(os.path.join(output_path,"rotated180_{}.png".format(distance_metric)),rotated180)
	rotated270 = main_SR(rotated270,distance_metric)
	cv2.imwrite(os.path.join(output_path,"rotated270_{}.png".format(distance_metric)),rotated270)

	flipped = main_SR(flipped,distance_metric)
	cv2.imwrite(os.path.join(output_path,"flipped_{}.png".format(distance_metric)),flipped)
	rotated90_flipped = main_SR(rotated90_flipped,distance_metric)
	cv2.imwrite(os.path.join(output_path,"rotated90_flipped_{}.png".format(distance_metric)),rotated90_flipped)
	rotated180_flipped = main_SR(rotated180_flipped,distance_metric)
	cv2.imwrite(os.path.join(output_path,"rotated180_flipped_{}.png".format(distance_metric)),rotated180_flipped)
	rotated270_flipped = main_SR(rotated270_flipped,distance_metric)
	cv2.imwrite(os.path.join(output_path,"rotated270_flipped_{}.png".format(distance_metric)),rotated270_flipped)

	# Apply the inverse Transformation to the High Resolution images.
	rotated90_back = cv2.rotate(rotated90, cv2.ROTATE_90_COUNTERCLOCKWISE)
	rotated180_back = cv2.rotate(rotated180, cv2.ROTATE_180)	
	rotated270_back = cv2.rotate(rotated270, cv2.ROTATE_90_CLOCKWISE)	
	flipped_back = cv2.flip(flipped,0)
	rotated90_flipped_back = cv2.rotate(cv2.flip(rotated90_flipped,0), cv2.ROTATE_90_COUNTERCLOCKWISE ) 
	rotated180_flipped_back = cv2.rotate(cv2.flip(rotated180_flipped,0), cv2.ROTATE_180 ) 
	rotated270_flipped_back = cv2.rotate(cv2.flip(rotated270_flipped,0), cv2.ROTATE_90_CLOCKWISE )
	
	# Change the datatype of the numpy array so that it will not do clipping on adding two images.
	orig = orig.astype('int')
	rotated90_back = rotated90_back.astype('int')
	rotated180_back = rotated180_back.astype('int')
	rotated270_back = rotated270_back.astype('int')
	flipped_back = flipped_back.astype('int')
	rotated90_flipped_back = rotated90_flipped_back.astype('int')
	rotated180_flipped_back = rotated180_flipped_back.astype('int')
	rotated270_flipped_back = rotated270_flipped_back.astype('int')
	

	# Take the average of all output images.
	avg_img = (orig + rotated90_back + rotated180_back + rotated270_back + flipped_back + rotated90_flipped_back + rotated180_flipped_back + rotated270_flipped_back)//8
	avg_img = avg_img.astype('uint8')

	cv2.imwrite(os.path.join(output_path,"SR_paper_{}_enhanced_p.png".format(distance_metric)),avg_img)
	print("\nDONE !!!!\n\n")