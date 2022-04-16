import cv2
import os
from utils import *
from const import *
import pyflann
#from scipy.spatial import distance


"""
Rough Work:

INTER_NEAREST



"""
distance_metric = 'euclidean'
def SR(img):

	img_pyr = build_pyramid(img)
	highest_lvl_filled = MID;
	out_x, out_y = translate_img_by_half_pixel(img)
	cv2.imwrite("out_x.png",out_x)
	cv2.imwrite("out_y.png",out_y)


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
	#NNs, Dist = knnsearch(patches_db, input_patches_p,K)
	# pyflann.set_distance_type(distance_type='euclidean')
	# flann = pyflann.FLANN()

	#NNs, Dist = flann.nn(patches_db, input_patches_p,K, algorithm="kmeans")
	#NNs, Dist = kneareset_neighbour(patches_db, input_patches_p, k=K, distance_metric = 'euclidean')
	#NNs, Dist = knnsearch_cosine(patches_db, input_patches_p_p,K)
	#NNs, Dist = knnsearch_new(patches_db, input_patches_p,K)
	#NNs, Dist = kneareset_neighbour_scipy_cosine(patches_db, input_patches_p,k=K)
	NNs, Dist = knnsearch_scikit(patches_db, input_patches_p,k=K, custom_distance_metric=distance_metric)
	#Dist = np.sqrt(Dist)
	print("nn = {} , D(0,0) {} ".format(NNs[0,0],Dist[0,0]) )
	print("nn = {} , D(0,1) {} ".format(NNs[0,1],Dist[0,1]) )
	print("nn = {} , D(0,2) {} ".format(NNs[0,2],Dist[0,2]) )

	print("len(NNs) = ",len(NNs))
	print("len(NNs[0]) = ",len(NNs[0]))
	print("type(NNs)",type(NNs))
	print("type(NNs[0])",type(NNs[0]))
	print("NNs.shape",NNs.shape)

	for next_target in range(next_target_start,NUMCELLS+1):
		skipped = 0
		skipped_no_info=0
		delta = next_target-MID
		(htg,wtg) = img_pyr[next_target].shape	

		new_img = np.ones((htg,wtg))*DEFAULT_BG_GREYVAL
		weighted_dists = np.zeros((htg,wtg))
		sum_weights = np.zeros((htg,wtg))
		factor_src = htg/img.shape[0]

		print("factor_src = ",factor_src)
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

		print("shape of weighted_dists = ", weighted_dists.shape)
		print("shape of sum_weights = ", sum_weights.shape)
		new_img = weighted_dists/sum_weights
		new_img[ np.isnan(new_img) ] = 0

		## DO UNSHARP MASKING here.....

		# cv2.imshow("output",new_img)
		# cv2.waitKey()
		# cv2.destroyAllWindows()
		img_pyr[next_target] = new_img
		highest_lvl_filled = next_target


	output = new_img
	return output




if __name__ == "__main__":

	images = ["inp1_forest.png","inp2_world_war2.png", "inp3_building.png"]
	#output_resolutions = [ (552, 296), (610, 385), (487,261)]
	#output_resolutions = [ (296,552), (385,610), (261,487)]

	for i in range(len(images)):
	
		# i = 2
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

		print(grey_img)
		# print("after = ",np.max(grey_img))
		# print("after = ",np.min(grey_img))


		print(grey_img.shape)

		grey_img = grey_img/255
		grey_SRed = SR(grey_img)


		grey_SRed *= 255
		#cv2.imwrite(images[i][:-4]+"_grey.png",grey_SRed)
		yiq_img_big[:,:,0] = grey_SRed

		img_SRed = transformYIQ2RGB(yiq_img_big)
		img_SRed = img_SRed.astype(np.uint8)
		img_SRed = cv2.cvtColor(img_SRed, cv2.COLOR_RGB2BGR)
		(h,w,_) = img_SRed.shape
		img_SRed = img_SRed[5:h-5, 5:w-5]
		img_SRed = cv2.resize(img_SRed,(w,h))

		cv2.imwrite(images[i][:-4]+"_SR_paper_{}.png".format(distance_metric),img_SRed)
