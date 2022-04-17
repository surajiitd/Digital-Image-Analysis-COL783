
import numpy as np
import maxflow
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_graph(image):
    g = maxflow.Graph[float]()
    img = image
    
    (h,w,c) = img.shape
    
    #converting image to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dx = np.diff(img_gray.astype(np.int64),axis=1)
    dx = np.concatenate((dx,dx[:,-1].reshape(-1,1)),axis=1)
    dy = np.diff(img_gray.astype(np.int64),axis=0)
    dy = np.concatenate((dy,dy[-1,:].reshape(1,-1)),axis=0)
    e1 = np.absolute(dx) + np.absolute(dy)
    e1 = e1.astype(np.uint8)

    nodeids = g.add_grid_nodes((h,w))

    # Edges pointing backwards (left, left up and left down) with infinite
    # capacity
    structure = np.array(
        [[np.inf, 0, 0],
         [np.inf, 0, 0],
         [np.inf, 0, 0]]
    )
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)

    weights = e1
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

number_of_vertical_seams_removed = int(0.25*width)
print("original image shape is ",original_image.shape)
print("Total seams to be removed is ",number_of_vertical_seams_removed)


for x in range(number_of_vertical_seams_removed):
    print("Removing the seam ",x)
    print("------------------------")
    
    #img = cv2.imread(frames_path+'frame0.jpg')
    height,width,c = img.shape

    nodeids, g = create_graph(img)
    g.maxflow()
    op=g.get_grid_segments(nodeids)
    
    
    seam = []
    
    for i in range(height):
        for j in range(width):
            if op[i][j] == True:
                seam.append(j)
                break
    
    dup_img = img.copy()
    
    
    for i in range(height):
        dup_img = cv2.circle(dup_img,(seam[i],i),1,(0,0,255),1)
        dup_img_2 = cv2.circle(dup_img_2,(seam[i],i),1,(0,0,255),1)
    cv2.imwrite(os.path.join(output_path,"verti"+str(x)+".jpg"), dup_img)

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
    print("-------------------------")
    #break
        
cv2.imshow("input_image",original_image)
cv2.imshow("output_image",img)
cv2.imshow("allseams_image",dup_img_2)
cv2.imwrite(os.path.join(output_path,"final_image.jpg"), img)
cv2.imwrite(os.path.join(output_path,"final_image_all_seams.jpg"), dup_img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()