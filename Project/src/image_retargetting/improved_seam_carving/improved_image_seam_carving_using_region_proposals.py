
import numpy as np
import maxflow
import cv2
import sys
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def non_max_suppression_fast(boxes, overlapThresh=0.5):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def region_proposal(im):
    
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    orig_im = im.copy()
    orig_height,orig_width,c = im.shape
    newHeight = 200
    newWidth = int(im.shape[1]*200/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight)) 
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    
    operation = 'q'
    
    if (operation == 'f'):
        ss.switchToSelectiveSearchFast()
    
    elif (operation == 'q'):
        ss.switchToSelectiveSearchQuality()
    else:
        sys.exit(1)
        
    rects = ss.process()
    
    numShowRects = 10
    
    increment = 50
    boxes = []
    imOut = im.copy()
    
    
    for i, rect in enumerate(rects):
        #print(i)
        if (i < numShowRects):
            x, y, w, h = rect
            boxes.append([x,y,x+w,y+h])
        else:
            break
    
    
    boxes = np.array(boxes)
    final_boxes = non_max_suppression_fast(boxes)

    mask_img = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
    
    for bbox in final_boxes:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0),1)
        cv2.rectangle(mask_img, (x1, y1), (x2,y2), (255, 255, 255),-1)
        
    mask_img = cv2.resize(mask_img, (orig_width, orig_height))
    
    
    return mask_img

def create_graph(image,mask_img):
    g = maxflow.Graph[float]()   
    img = image
    
    (h,w,c) = img.shape
    
    #converting image to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)
    
    dx = np.diff(img_gray.astype(np.int64),axis=1)
    dx = np.concatenate((dx,dx[:,-1].reshape(-1,1)),axis=1)
    dy = np.diff(img_gray.astype(np.int64),axis=0)
    dy = np.concatenate((dy,dy[-1,:].reshape(1,-1)),axis=0)
    e1 = np.absolute(dx) + np.absolute(dy)
    e1 = e1.astype(np.uint8)
    #dest_and = cv2.bitwise_and(mask_img, e1, mask = None)
    addition = cv2.add(mask_img,e1)
    

    
    nodeids = g.add_grid_nodes((h,w))

    # Edges pointing backwards (left, left up and left down) with infinite
    # capacity
    structure = np.array(
        [[np.inf, 0, 0],
         [np.inf, 0, 0],
         [np.inf, 0, 0]]
    )
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)

    # Set a few arbitrary weights
    #weights = np.array([[100, 110, 120, 130, 140]]).T + np.array([0, 2, 4, 6, 8])
    weights = addition
    #print(weights)

    # Edges pointing right
    structure = np.zeros((3, 3))
    structure[1, 2] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # Source node connected to leftmost non-terminal nodes.
    left = nodeids[:, 0]
    g.add_grid_tedges(left, np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    right = nodeids[:, -1]
    g.add_grid_tedges(right, 0, np.inf)

    return nodeids, g

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inp', type=str, help='path to the input image')
parser.add_argument('-o','--out', type=str, help='path to store output')
parser.add_argument('-n','--img_no', type=str, help='input image number')
args = parser.parse_args()

img_number = args.img_no
image_path = args.inp  #"D:\\IIT DELHI CLASSES\\DIGITAL IMAGE ANALYSIS\\Assignments\\project\\Project_dataset\\Project_dataset\\Image_retargeting\\Taks_1_dataset\\"
output_path = args.out  #"D:\\IIT DELHI CLASSES\\DIGITAL IMAGE ANALYSIS\\Assignments\\project\\final_programs\\image_retargetting\\without_forward_energy\\output"+str(img_number)+"\\"
output_path = os.path.join(output_path, "output"+str(img_number))
make_dir(output_path)

img = cv2.imread(image_path)

original_image = img.copy()
dup_img_2 = original_image.copy()
height,width,c = img.shape

number_of_seams_removed = int(0.25*width)
#number_of_seams_removed = 10
print("original image shape is ",original_image.shape)
print("Total seams to be removed is ",number_of_seams_removed)

mask_img = region_proposal(img)
cv2.imwrite(output_path+"final_mask_image.jpg", mask_img)

for x in range(number_of_seams_removed):
    print(x)
    dup_img = img.copy()
    height,width,c = img.shape
    nodeids, g = create_graph(img,mask_img)
    
    seam = []
    flow = g.maxflow()
    op = g.get_grid_segments(nodeids)

    for i in range(len(op)):
        for j in range(len(op[0])):
            if op[i][j] == True:
                seam.append(j)
                break

    
    for i in range(height):
        dup_img = cv2.circle(dup_img,(seam[i],i),1,(0,0,255),1)
        dup_img_2 = cv2.circle(dup_img_2,(seam[i],i),1,(0,0,255),1)
    cv2.imwrite(os.path.join(output_path,str(x)+".jpg"), dup_img)

    
    for i in range(height):
        j_ind = seam[i]
        if j_ind+1 >= width:
            img[i,:-1] = img[i,:-1]
        elif j_ind == 0:
            img[i,:-1] = img[i,1:]
        else:
            img[i,j_ind:-1] = img[i,j_ind+1:]
    #j_ind +=1
    img = img[:,:-1]
    
    for i in range(height):
        j_ind = seam[i]
        if j_ind+1 >= width:
            mask_img[i,:-1] = mask_img[i,:-1]
        elif j_ind == 0:
            mask_img[i,:-1] = mask_img[i,1:]
        else:
            mask_img[i,j_ind:-1] = mask_img[i,j_ind+1:]
    #j_ind +=1
    mask_img = mask_img[:,:-1]
    
        
cv2.imshow("input_image",original_image)
cv2.imshow("output_image",img)
cv2.imwrite(os.path.join(output_path,"final_image.jpg"), img)

cv2.imwrite(os.path.join(output_path,"all_seam_image.jpg"), dup_img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    #print(g.get_grid_segments(nodeids))