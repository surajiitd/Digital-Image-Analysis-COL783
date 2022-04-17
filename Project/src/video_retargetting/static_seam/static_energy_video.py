
import numpy as np
import maxflow
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse

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
    
    
    global_energy = (0.3*spatial_energy)+(0.7*temporal_energy)
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

            
