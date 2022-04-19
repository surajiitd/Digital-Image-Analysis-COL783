import os
import cv2
import numpy as np
import sys 

def readImages(path):
    image_list = []
    image_names = os.listdir(image_directory_path)
    image_names.sort()
    for i in range(len(image_names)):
        width = 640
        height = 480
        input_image = cv2.imread(os.path.join(path, image_names[i]))
        input_image = cv2.resize(input_image, (width, height))
        image_list.append(input_image)
    return image_list
    

def findAndDescribeKeyPoints(image):

    descriptor = cv2.xfeatures2d.SIFT_create()
    (keypoints, features) = descriptor.detectAndCompute(image, None)

    points=[]
    for kp in keypoints:
        points.append(kp.pt)
    
    keypoints = np.float32(points)
    
    return (keypoints, features)

def featureMatching(src,tgt,transformation="affine"):

    ratio = 0.75
    reprojThresh = 4
    (kpsA, featuresA) = findAndDescribeKeyPoints(src)
    (kpsB, featuresB) = findAndDescribeKeyPoints(tgt)

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
        
    if len(matches) > 4:
    
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        
        if transformation == "projective":
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            return H
            
        elif transformation == "affine":
            mat =  cv2.estimateAffinePartial2D(ptsA,ptsB)[0]
            mat = np.vstack([mat,[0.0,0.0,1.0]])
            return mat
        
    return np.eye(3)

def performWarp(src, homography_list,output_image, y_offset, x_offset):


    output_h, output_w, output_c = output_image.shape
    input_h, input_w, input_c = src.shape

    if homography_list is None:
        
        end_y = y_offset+input_h
        end_x = x_offset+input_w
        output_image[y_offset:end_y, x_offset:end_x] = src
    
    else:

        homography = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        
        i = 0
        
        while i < len(homography_list):
            homography = homography_list[i]@homography
            i+=1

        # Finding bounding box
        edge_points = np.array([[0, 0, 1], [input_w, input_h, 1],[input_w, 0, 1], [0, input_h, 1]])
        edge_points = edge_points.T
        
        borders = homography@edge_points.reshape(3, -1)
        borders = borders.reshape(edge_points.shape)
        scale = borders[-1]
        borders = borders/scale
        
        borders = (borders+np.array([x_offset, y_offset, 0])[:, np.newaxis]).astype(int)
        
        height_min = np.min(borders[1])
        height_max = np.max(borders[1])
        width_min = np.min(borders[0])
        width_max = np.max(borders[0])

        # Filling the bounding box in imgout
        homography_inverse = np.linalg.inv(homography)
        
        i = height_min
        while i <= height_max:
            j = width_min
            while j <= width_max:
                
                if (0 <= i < H):
                    if (0 <= j < W):
                        projected_y = i - y_offset
                        projected_x = j - x_offset
                        t = np.array([projected_x, projected_y, 1])
                        actual_matrix = homography_inverse@t
                        actual_j,actual_i,actual_scale = actual_matrix
                        actual_i,actual_j = int(actual_i/actual_scale),int(actual_j/actual_scale)
                        
                        if(0 <= actual_i < input_h): 
                            if (0 <= actual_j < input_w):
                                output_image[i, j] = src[actual_i, actual_j]
                j+=1
            i+=1


    mask = np.sum(output_image, axis=2).astype(bool)
    return output_image, mask

def laplacianBlending(images, masks, n=5):

    gaussian_pyramids = {}
    laplacian_pyramids = {}
    
    temp = images[0].shape
    H = temp[0]
    W = temp[1]
    C = temp[2] 

    i = 0
    while i < len(images):

        
        gaussian_pyramids[i] = [images[i].copy()]
        gp = images[i].copy()
        x = 0
        while x < n:
            gp = cv2.pyrDown(gp)
            gaussian_pyramids[i].append(gp)
            x+=1

        laplacian_pyramids[i] = [gp]
        j = len(gaussian_pyramids[i])-2
        while j >=0:
            gp_up = cv2.pyrUp(gp)
            gp = gaussian_pyramids[i][j]
            lp = cv2.subtract(gp, gp_up)
            laplacian_pyramids[i].append(lp)
            j-=1
        i+=1

    default_mask = masks[0].copy()
    default_image = images[0].copy()
    default_pyramids = []
    for i in range(len(laplacian_pyramids[0])):
        default_pyramids.append(laplacian_pyramids[0][i].copy())

    bleanded_image = None
    i = 1
    
    while i < len(images):

        y1, x1 = np.where(default_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        right_corner_image_1 = np.max(x1)
        right_corner_image_2 = np.max(x2)
        
        if right_corner_image_1 > right_corner_image_2:
            left_pyramid = laplacian_pyramids[i]
            right_pyramid = default_pyramids
            
        if right_corner_image_1 < right_corner_image_2:
            left_pyramid = default_pyramids
            right_pyramid = laplacian_pyramids[i]

        # To check if the two pictures need to be blended are overlapping or not
        mask_inter = np.bitwise_and(default_mask, masks[i])

        if True in mask_inter:
            y, x = np.where(mask_inter == 1)
            x_min = np.min(x)
            x_max = np.max(x)

            split_range = (x_max-x_min)/2
            split = (split_range + x_min)/W

            blended_images_list = []
            for la, lb in zip(left_pyramid, right_pyramid):
                t = la.shape
                rows = t[0]
                cols = t[1]
                dpt = t[2]
                
                center_limit = int(split*cols)
                ls = np.hstack((la[:, 0:center_limit], lb[:, center_limit:]))
                blended_images_list.append(ls)

        else:
            print("hey there")
            blended_images_list = []
            for la, lb in zip(left_pyramid, right_pyramid):
                t = la.shape
                rows = t[0]
                cols = t[1]
                dpt = t[2]
                ls = la + lb
                blended_images_list.append(ls)

        # Reconstructing the image
        bleanded_image = blended_images_list[0]
        j=1
        while j <= n:
            bleanded_image = cv2.pyrUp(bleanded_image)
            bleanded_image = cv2.add(bleanded_image, blended_images_list[j])
            j+=1

        # Preparing the commong image for next image to be added
        default_image = bleanded_image
        default_mask = np.sum(default_image.astype(bool), axis=2).astype(bool)
        default_pyramids = blended_images_list
        i+=1

    return bleanded_image



def feathering(images, masks):

    temp = images[0].shape
    H = temp[0]
    W = temp[1]
    C = temp[2] 


    default_mask = masks[0].copy()
    default_image = images[0].copy()


    ls = None
    for i in range(1, len(images)):

        y1, x1 = np.where(default_mask == 1)
        y2, x2 = np.where(masks[i] == 1)
        
        blending_image = images[i].copy()
        
        right_corner_image_1 = np.max(x1)
        right_corner_image_2 = np.max(x2)


        if right_corner_image_1 > right_corner_image_2:
            left_py = blending_image
            right_py = default_image

        if right_corner_image_1 < right_corner_image_2:
            left_py = default_image
            right_py = blending_image

        mask_intersection = np.bitwise_and(default_mask, masks[i])

        if True in mask_intersection:

            y, x = np.where(mask_intersection == 1)
            x_min = np.min(x)
            x_max = np.max(x)

            split_range = (x_max-x_min)/2
            split = (split_range + x_min)/W


            rows, cols, dpt = left_py.shape
            ls = np.hstack((left_py[:, 0:int(split*cols)], right_py[:, int(split*cols):]))
            ls = cv2.GaussianBlur(ls,(3,3),0)

        else:

            rows, cols, dpt = left_py.shape
            ls = left_py + right_py



        # Preparing the commong image for next image to be added
        default_image = ls
        default_mask = np.sum(default_image.astype(bool), axis=2).astype(bool)

    return ls

def poisson_blending(img_outputs, masks):
    panorma = img_outputs[0]

    cv2.imwrite("panoroma.jpg",panorma)

    for i in range(1,len(img_outputs)):
        obj= img_outputs[i]
        cv2.imwrite("obj.jpg",obj)
        img_mask = masks[i]
        obj_mask = np.array(mask) * 255

        
        print("\nPoisson Blending: \n")


        cv2.imwrite("mymask.jpg",obj_mask)
        obj_mask = cv2.imread("mymask.jpg",cv2.IMREAD_GRAYSCALE)

        y, x = np.where(obj_mask==255)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        center = ((x_min+x_max)//2, (y_min+y_max)//2 ) 

        print("center coordinate:",center)

        cropped_obj = obj[y_min:y_max, x_min:x_max]
        cropped_obj_mask = obj_mask[y_min:y_max, x_min:x_max]

        # Seamlessly clone src into dst and put the results in output
        print(panorma.shape)
        print(cropped_obj.shape)
        print(cropped_obj_mask.shape)
        cv2.imwrite("cropped_obj.jpg",cropped_obj)
        cv2.imwrite("cropped_obj_mask.jpg",cropped_obj_mask)

        print("center =",center)
        normal_clone = cv2.seamlessClone(cropped_obj, panorma, cropped_obj_mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite("opencv-normal-clone-example.jpg", normal_clone)


image_directory_path = sys.argv[1]
# ex_no = input("which image 1,2,3,4?:")
# image_directory_path = "../data/{}".format(ex_no)

transformation = "projective" # "affine" or "projective"                         
blend_method = "feathering" # "feathering" or "pyramid"


image_names = os.listdir(image_directory_path)
image_names.sort()
n = len(image_names)    
print("No. of images to mosaic:",n)
increase_panoroma_height_by = 2                          

Images = readImages(image_directory_path)

img_shape = np.array(Images[0].shape)
H = img_shape[0]*increase_panoroma_height_by
W = img_shape[1]*n
C = img_shape[2]*1     


final_image = np.zeros((H, W, C))
output_images = []
all_masks = []

central_image = Images[n//2]
x_offsets = W//2
y_offset = H//2
img, mask = performWarp(central_image, None, final_image.copy(), y_offset, x_offsets)

output_images.append(img)
all_masks.append(mask)
left_homography = []
right_homography = []

i = 1
half_length = len(Images)//2

while i <= half_length:

    # right
    if (n//2+i < n) and ((n//2+i-1) < n):  
        print(f"\n image {n//2+i} ")
        left_image = Images[n//2+(i-1)]
        right_image = Images[n//2+i]
        matrix = featureMatching(right_image, left_image,transformation)
        right_homography.append(matrix)
        img, mask = performWarp(right_image, right_homography,final_image.copy(), y_offset, x_offsets)
        output_images.append(img)
        all_masks.append(mask)
        
    if (n//2-i >= 0) and ((n//2+i-1) >= 0):
        print(f"\n image {n//2-i} ")
        left_image = Images[n//2-(i-1)]
        right_image = Images[n//2-i]
        matrix = featureMatching(right_image, left_image,transformation)
        left_homography.append(matrix)
        img, mask = performWarp(right_image, left_homography,final_image.copy(), y_offset, x_offsets)
        output_images.append(img)
        all_masks.append(mask)
    
    i+=1


if blend_method=="feathering":
    uncropped = feathering(output_images, all_masks)
elif blend_method=="pyramid" :
    uncropped = laplacianBlending(output_images, all_masks)
#poisson_blending(output_images, all_masks)

mask = np.sum(uncropped, axis=2).astype(bool)

yy, xx = np.where(mask == 1)
x_min = np.min(xx)
x_max = np.max(xx)
y_min = np.min(yy)
y_max = np.max(yy)

final = uncropped[y_min:y_max, x_min:x_max]

cv2.imwrite("Mosaiced_Image_{}_{}.jpg".format(transformation, blend_method), final)

