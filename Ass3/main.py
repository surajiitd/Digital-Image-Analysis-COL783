import cv2
import os
from utils import *
from const import *
import pyflann

"""
Rough Work:

INTER_NEAREST



"""

def SR(grey_img):

	img_pyr = build_pyramid(grey_img)
	highest_lvl_filled = MID;
	out_x, out_y = translate_img_by_half_pixel(grey_img)
	cv2.imwrite("out_x.png",out_x)
	cv2.imwrite("out_y.png",out_y)


	#Init patch database, according to current highest filled layer.
	patches_db = np.array([-1]*PATCH_SIZE,dtype=np.float32)
	qi = np.array([])
	qj = np.array([])
	qlvl = np.array([])
	for lvlq in range(MID-1,0,-1):
		(input_patches_q,lvlq_pi, lvlq_pj) = img2patches(img_pyr[lvlq])
		patches_db = np.vstack([patches_db, input_patches_q])
		qi = np.append(qi,lvlq_pi)
		qj = np.append(qj,lvlq_pj)
		# patches_db.extend(input_patches_q)
		# qi.extend(lvlq_pi)
		# qj.extend(lvlq_pj)
		# levels = [lvlq for _ in range(len(lvlq_pi))]
		# qlvl.extend(levels)

		levels = np.ones((len(lvlq_pi)))   
		qlvl = np.append(qlvl, levels)
	patches_db = np.delete(patches_db,0,0)
	(input_patches_p,input_pi,input_pj) = img2patches(img_pyr[MID])

	print("LENGTH of 1 PATCH = ",len(input_patches_p[0]))
	next_target_start = MID+1;
	tot_numPatches = len(patches_db)
	query_numPatches = len(input_patches_p)
	print("Total no. of patches = ",tot_numPatches)
	print("no. of query patches = ",query_numPatches)

	pyflann.set_distance_type(distance_type='euclidean')
	flann = pyflann.FLANN()
	NNs, Dist = flann.nn(patches_db, input_patches_p,K)
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
		for p_idx in range(0,no_of_query_patches+1):
			if(p_idx % 1000 == 0):
				print("progress: patch {}/{}".format(p_idx,no_of_query_patches))


			pknns = NNs[p_idx]
			# check if the patch nn passes the min thresh criteria
			t_x = thresh( img, out_x, input_pi[p_idx],input_pj[p_idx] );
			t_y = thresh( img, out_y, input_pi[p_idx],input_pj[p_idx] );
			t = (t_x+t_y)/2

			#taken is counts the amount of predictions for current lr example which were set
			taken  = 0
			for k in range(0,K):

				nn = pknns[k]

				#parent patch
				imp = img_pyr[qlvl[nn] + delta]
				parent_h = imp.shape[0]

				#matched nearest neighbour
				imq = img_pyr[qlvl[nn]]
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
					(weighted_dists, sum_weights, new_img) = set_parent(img,input_pi(p_idx), input_pj(p_idx), new_img,hr_example, factor_src,weighted_dists,sum_weights, lr_patch,patches_db(nn,:) ) 

		new_img = weighted_dists/sum_weights
		new_img[ ]
		new_img = imsh...
		cv2.imshow("output",new_img)
		cv2.waitKey()
		cv2.destroyAllWindows()
		img_pyr[next_target] = new_img
		highest_lvl_filled = next_target


	output = new_img
	return output




if __name__ == "__main__":

	images = ["inp1_forest.png","inp2_world_war2.png", "inp3_building.png"]
	#output_resolutions = [ (552, 296), (610, 385), (487,261)]
	#output_resolutions = [ (296,552), (385,610), (261,487)]

	#for i in range(len(images)):
	#will make loop later for all 3
	i = 0
	img = cv2.imread(os.path.join("Assignment3_data",images[i]) )
	# cv2.imshow("img",img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

	output_resolution = (img.shape[1]*SCALE,img.shape[0]*SCALE)

	#Nearest Neighbor Interpolation Method
	img_nearest = cv2.resize(img,output_resolution, interpolation=cv2.INTER_NEAREST)
	cv2.imwrite(images[i][:-4]+"_nearest.png", img_nearest)

	#BiCubic Interpolation Method
	img_bicubic = cv2.resize(img,output_resolution, interpolation=cv2.INTER_CUBIC)
	cv2.imwrite(images[i][:-4]+"_bicubic.png", img_bicubic)


	yiq_img = transformRGB2YIQ(img_bicubic)

	grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print(grey_img.shape)
	grey_SRed = SR(grey_img)
	# yiq_img[:,:,0] = grey_SRed

	# img_SRed = transformYIQ2RGB(yiq_img)
	# imwrite(images[i][:-4]+"_SR_paper.png",img_SRed)