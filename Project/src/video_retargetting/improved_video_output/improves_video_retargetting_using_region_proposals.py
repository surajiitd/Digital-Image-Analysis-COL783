
import numpy as np
import maxflow
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import argparse

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
    newHeight = 50
    newWidth = int(im.shape[1]*50/im.shape[0])
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
    
    numShowRects = 5
    
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

def FrameCapture(path,output_path):
      

    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
  
    while success:
        
        success, image = vidObj.read()
        if not success:
            break
        
        cv2.imwrite(os.path.join(output_path,"frame%d.jpg" % count), image)
        count += 1

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_static_energy(path,direction="vertical"):
    frames_path = path
    frames_count = len(os.listdir(path))
    spatial_frame_energy_list = []
    temporal_frame_energy_list = []
    region_proposal_energy_list = []
    
    
    for i in range(frames_count):
    
        img = cv2.imread(os.path.join(frames_path,"frame"+str(i)+".jpg"))
        
        if direction == "horizontal":
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        
        height,width,channel = img.shape
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        dx = np.diff(img_gray.astype(np.int64),axis=1)
        dx = np.concatenate((dx,dx[:,-1].reshape(-1,1)),axis=1)
        dy = np.diff(img_gray.astype(np.int64),axis=0)
        dy = np.concatenate((dy,dy[-1,:].reshape(1,-1)),axis=0)
        e1 = np.absolute(dx) + np.absolute(dy)
        e1 = e1.astype(np.uint8)
        
        spatial_frame_energy_list.append(e1)
    
    spatial_frame_energy_list = np.array(spatial_frame_energy_list)
    spatial_energy = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            max_pixel_value = np.amax(spatial_frame_energy_list[:,i,j])
            #spatial_energy[i,j]= max_pixel_value/255
            spatial_energy[i,j]= max_pixel_value
    
    
    
    for i in range(1,frames_count):
    
        img1 = cv2.imread(os.path.join(frames_path,"frame"+str(i-1)+".jpg"))
        img2 = cv2.imread(os.path.join(frames_path,"frame"+str(i)+".jpg"))
        
        if direction == "horizontal":
            img1 = cv2.rotate(img1, cv2.cv2.ROTATE_90_CLOCKWISE)
            img2 = cv2.rotate(img2, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        height,width,channel = img1.shape
        
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(img2_gray, img1_gray)
        
        temporal_frame_energy_list.append(diff)
    
    
    temporal_frame_energy_list = np.array(temporal_frame_energy_list)
    temporal_energy = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            max_pixel_value = np.amax(temporal_frame_energy_list[:,i,j])
            temporal_energy[i,j]= max_pixel_value
    
            
    for i in range(frames_count):
    
        img = cv2.imread(os.path.join(frames_path,"frame"+str(i)+".jpg"))
        
        if direction == "horizontal":
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        
        height,width,channel = img.shape
        mask_img = region_proposal(img)
        region_proposal_energy_list.append(mask_img)
    
    region_proposal_energy_list = np.array(region_proposal_energy_list)
    region_proposal_energy = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            max_pixel_value = np.amax(region_proposal_energy_list[:,i,j])
            #spatial_energy[i,j]= max_pixel_value/255
            region_proposal_energy[i,j]= max_pixel_value
    
    
    global_energy = (0.3*spatial_energy)+(0.4*temporal_energy)+(0.3*region_proposal_energy)
    
    #cv2.imshow("global_energy",global_energy/255)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return global_energy


def create_graph(h,w,static_energy):
    g = maxflow.Graph[float]()


    energy = static_energy

    nodeids = g.add_grid_nodes((h,w))

    structure = np.array(
        [[np.inf, 0, 0],
         [np.inf, 0, 0],
         [np.inf, 0, 0]]
    )
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)
    weights = energy


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
parser.add_argument('-i', '--inp', type=str, help='path to the input video')
parser.add_argument('-o','--out', type=str, help='path to store output')
parser.add_argument('-n','--vid_no', type=str, help='input video number')
args = parser.parse_args()


vid_number = args.vid_no
video_path = args.inp  #"D:\\IIT DELHI CLASSES\\DIGITAL IMAGE ANALYSIS\\Assignments\\project\\Project_dataset\\Project_dataset\\video_retargeting\\Task2_dataset\\"
frames_path = os.path.join(args.out,"video_"+str(vid_number))  #"D:\\IIT DELHI CLASSES\\DIGITAL IMAGE ANALYSIS\\Assignments\\project\\final_programs\\video_retargetting\\frame_wise_forward_energy\\video_"+str(vid_number)+"\\"
print(frames_path)
make_dir(frames_path)


FrameCapture(video_path,frames_path)

output_frame_path = os.path.join(args.out, "video_"+str(vid_number)+"_output")  # "D:\\IIT DELHI CLASSES\\DIGITAL IMAGE ANALYSIS\\Assignments\\project\\final_programs\\video_retargetting\\frame_wise_forward_energy\\video_"+str(vid_number)+"_output\\"
make_dir(output_frame_path)

frames_count = len(os.listdir(frames_path))
img = cv2.imread(os.path.join(frames_path,'frame0.jpg'))
height,width,c = img.shape

number_of_vertical_seams_removed = int(0.25*width)
print("original image shape is ",img.shape)
print("********************")
print("Removing vertical seams")
print("Total vertical seams to be removed is ",number_of_vertical_seams_removed)
print("********************")

for x in range(number_of_vertical_seams_removed):
    
    print("Removing the seam ",x)
    print("------------------------")
    static_energy = find_static_energy(frames_path,"vertical")
    temp_img = cv2.imread(os.path.join(frames_path,'frame0.jpg'))
    height,width,c = temp_img.shape
    nodeids, g = create_graph(height,width,static_energy)
    
    seam = []
    flow = g.maxflow()
    op = g.get_grid_segments(nodeids)

    for i in range(len(op)):
        for j in range(len(op[0])):
            if op[i][j] == True:
                seam.append(j)
                break
    
    for frame_number in range(frames_count):
        print("Working on the frame ",frame_number)
        img = cv2.imread(os.path.join(frames_path,'frame'+str(frame_number)+'.jpg'))
        original_image = img.copy()
        height,width,c = img.shape
        dup_img = img.copy()

        if frame_number == 0:
            for i in range(height):
                dup_img = cv2.circle(dup_img,(seam[i],i),1,(0,0,255),1)
            cv2.imwrite(os.path.join(output_frame_path,str(x)+".jpg"), dup_img)
            
            
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
        
        cv2.imwrite(os.path.join(frames_path,"frame"+str(frame_number)+".jpg"),img)
    print("-------------------------")

print("*************************")
print("Removing horizontal seams")
number_of_horizontal_seams_removed = int(0.5*height)
print("Total horizontal seams to be removed is ",number_of_horizontal_seams_removed)
print("*************************")

for x in range(number_of_horizontal_seams_removed):
    
    print("Removing the seam ",x)
    print("------------------------")
    static_energy = find_static_energy(frames_path,"horizontal")
    temp_img = cv2.imread(os.path.join(frames_path,'frame0.jpg'))
    temp_img = cv2.rotate(temp_img, cv2.cv2.ROTATE_90_CLOCKWISE)
    height,width,c = temp_img.shape
    nodeids, g = create_graph(height,width,static_energy)
    
    seam = []
    flow = g.maxflow()
    op = g.get_grid_segments(nodeids)

    for i in range(len(op)):
        for j in range(len(op[0])):
            if op[i][j] == True:
                seam.append(j)
                break
    
    for frame_number in range(frames_count):
        print("Working on the frame ",frame_number)
        img = cv2.imread(os.path.join(frames_path,'frame'+str(frame_number)+'.jpg'))
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        original_image = img.copy()
        height,width,c = img.shape
        dup_img = img.copy()

        if frame_number == 0:
            for i in range(height):
                dup_img = cv2.circle(dup_img,(seam[i],i),1,(0,0,255),1)
            dup_img = cv2.rotate(dup_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(output_frame_path,"hori"+str(x)+".jpg"), dup_img)
            
            
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
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(frames_path,"frame"+str(frame_number)+".jpg"),img)
    print("-------------------------")

            
