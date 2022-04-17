import numpy as np
import maxflow
import networkx as nx
import matplotlib.pyplot as plt
import os 
import cv2
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

def slow_forward_energy_graph_cut(img):
    height = img.shape[0]
    width = img.shape[1]
    
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    energy = np.zeros((height, width))
    lr_final = np.zeros((height, width))
    ulu_final = np.zeros((height, width))
    llu_final = np.zeros((height, width))
    m = np.zeros((height, width))
    
    for i in range(1, height):
        for j in range(width):
            up = (i-1) % height
            down = (i+1) % height
            left = (j-1) % width
            right = (j+1) % width
    
            mU = m[up,j]
            mL = m[up,left]
            mR = m[up,right]
                
            lr = np.abs(I[i,right] - I[i,left])
            ulu = np.abs(I[up,j] - I[i,left])
            llu = np.abs(I[down,j] - I[i,left]) 
            

            lr_final[i,j] = lr
            ulu_final[i,j] = ulu
            llu_final[i,j] = llu
            
    return lr_final,ulu_final,llu_final

def find_2d_manifold_energy(path,direction="vertical"):
    
    frames_path = path
    frames_count = len(os.listdir(path))
    lr_energy_list = []
    ulu_energy_list = []
    llu_energy_list = []
    
    for i in range(frames_count):
    
        img = cv2.imread(os.path.join(frames_path,"frame"+str(i)+".jpg"))
        
        if direction == "horizontal":
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        
        height,width,channel = img.shape
        
        lr_final,ulu_final,llu_final = slow_forward_energy_graph_cut(img)

        lr_final = lr_final.astype(np.uint8)
        ulu_final = ulu_final.astype(np.uint8)
        llu_final = llu_final.astype(np.uint8)
        
        lr_energy_list.append(lr_final)
        ulu_energy_list.append(ulu_final)
        llu_energy_list.append(llu_final)
    
    lr_energy_list = np.array(lr_energy_list)
    ulu_energy_list = np.array(ulu_energy_list)
    llu_energy_list = np.array(llu_energy_list)
    
    return lr_energy_list,ulu_energy_list,llu_energy_list


def create_graph(frames_path,direction="vertical",width=3, height=3, depth=2):

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((depth, height, width))
    structure = np.array(
        [[[0, 0, 0],
          [0, 0, 1],
          [0, 0, 0]]]
    )
    
    lr_energy_list,ulu_energy_list,llu_energy_list = find_2d_manifold_energy(frames_path,direction)
    g.add_grid_edges(nodeids, structure=structure,weights = lr_energy_list,symmetric=False)
    
    structure = np.array(
        [[[0, 1, 0],
          [0, 0, 0],
          [0, 0, 0]]]
    )
    g.add_grid_edges(nodeids, structure=structure,weights = ulu_energy_list,symmetric=False)

    structure = np.array(
        [[[0, 0, 0],
          [0, 0, 0],
          [0, 1, 0]]]
    )
    g.add_grid_edges(nodeids, structure=structure,weights = llu_energy_list,symmetric=False)    
    
    structure = np.array(
        [[[np.inf, 0, 0],
          [np.inf, 0, 0],
          [np.inf, 0, 0]]]
    )
    g.add_grid_edges(nodeids, structure=structure,symmetric=False)

    # Source node connected to leftmost non-terminal nodes.
    g.add_grid_tedges(nodeids[:, :, 0], np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    g.add_grid_tedges(nodeids[:, :, -1], 0, np.inf)
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

frames_output_path = os.path.join(args.out, "video_"+str(vid_number)+"_output") #"D:\\IIT DELHI CLASSES\\DIGITAL IMAGE ANALYSIS\\Assignments\\project\\final_programs\\video_retargetting\\2d_manifold_output\\video_"+str(vid_number)+"_output\\"
make_dir(frames_output_path)

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
    
    img = cv2.imread(os.path.join(frames_path,'frame0.jpg'))
    height,width,c = img.shape

    nodeids, g = create_graph(frames_path,"vertical",width=width,height=height,depth=frames_count)
    g.maxflow()
    op=g.get_grid_segments(nodeids)
    
    all_seams =[]
    
    for frame in range(len(op)):
        seam = []
        for i in range(height):
            for j in range(width):
                if op[frame][i][j] == True:
                    seam.append(j)
                    break
        all_seams.append(seam)
    #print(all_seams[1])
    #break

    
    for frame in range(frames_count):
        print("Working on the frame ",frame)
        img = cv2.imread(os.path.join(frames_path,'frame'+str(frame)+'.jpg'))
        original_image = img.copy()
        height,width,c = img.shape
        dup_img = img.copy()
        
        if frame == 0:
            for i in range(height):
                dup_img = cv2.circle(dup_img,(all_seams[frame][i],i),1,(0,0,255),1)
            cv2.imwrite(os.path.join(frames_output_path,"verti"+str(x)+".jpg"), dup_img)
    
        for i in range(height):
            j_ind = all_seams[frame][i]
            if j_ind+1 >= width:
                img[i,:-1] = img[i,:-1]
            elif j_ind == 0:
                img[i,:-1] = img[i,1:]
            else:
                img[i,j_ind:-1] = img[i,j_ind+1:]
            #j_ind +=1
        img = img[:,:-1]
        cv2.imwrite(os.path.join(frames_path,"frame"+str(frame)+".jpg"),img)
    print("-------------------------")
    #break

print("*************************")
print("Removing horizontal seams")
number_of_horizontal_seams_removed = int(0.5*height)
print("Total horizontal seams to be removed is ",number_of_horizontal_seams_removed)
print("*************************")

for x in range(number_of_horizontal_seams_removed):
    
    print("Removing the seam ",x)
    print("------------------------")
    img = cv2.imread(os.path.join(frames_path,'frame0.jpg'))
    height,width,c = img.shape
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    height,width,c = img.shape
    
    nodeids, g = create_graph(frames_path,"horizontal",width=width,height=height,depth=frames_count)
    g.maxflow()
    op=g.get_grid_segments(nodeids)
    
    all_seams =[]
    
    for frame in range(len(op)):
        seam = []
        for i in range(height):
            for j in range(width):
                if op[frame][i][j] == True:
                    seam.append(j)
                    break
        all_seams.append(seam)
    
    for frame in range(frames_count):
        print("Working on the frame ",frame)
        img = cv2.imread(os.path.join(frames_path,'frame'+str(frame)+'.jpg'))
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        original_image = img.copy()
        height,width,c = img.shape
        dup_img = img.copy()

        if frame == 0:
            for i in range(height):
                dup_img = cv2.circle(dup_img,(all_seams[frame][i],i),1,(0,0,255),1)
            dup_img = cv2.rotate(dup_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(frames_output_path,"hori"+str(x)+".jpg"), dup_img)
            
            
        for i in range(height):
            j_ind = all_seams[frame][i]
            if j_ind+1 >= width:
                img[i,:-1] = img[i,:-1]
            elif j_ind == 0:
                img[i,:-1] = img[i,1:]
            else:
                img[i,j_ind:-1] = img[i,j_ind+1:]
        #j_ind +=1
        img = img[:,:-1]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(frames_path,"frame"+str(frame)+".jpg"),img)
    print("-------------------------")
