import os
import cv2
import numpy as np
import sys


def readImages(path):
    image_list = []
    image_names = os.listdir(image_directory_path)
    image_names.sort()
    for i in range(len(image_names)):
        img = cv2.imread(os.path.join(path, image_names[i]))
        img = cv2.resize(img, (640, 480))
        image_list.append(img)
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
    #print("matches")
    #print(matches)
        
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

def warp(src, homography, imgout, y_offset, x_offset):


    # Getting the shapes
    output_h, output_w, output_c = imgout.shape
    src_h, src_w, src_c = src.shape

    # Checking if image needs to be warped or not
    if homography is not None:

        # Calculating net homography
        t = homography
        homography = np.eye(3)
        for i in range(len(t)):
            homography = t[i]@homography

        # Finding bounding box
        pts = np.array([[0, 0, 1], [src_w, src_h, 1],
                        [src_w, 0, 1], [0, src_h, 1]]).T
        borders = (homography@pts.reshape(3, -1)).reshape(pts.shape)
        borders /= borders[-1]
        borders = (borders+np.array([x_offset, y_offset, 0])[:, np.newaxis]).astype(int)
        print("border is ")
        print(borders)
        h_min, h_max = np.min(borders[1]), np.max(borders[1])
        w_min, w_max = np.min(borders[0]), np.max(borders[0])

        # Filling the bounding box in imgout
        h_inv = np.linalg.inv(homography)
        for i in range(h_min, h_max+1):
            for j in range(w_min, w_max+1):

                if (0 <= i < H and 0 <= j < W):
                    # Calculating image cordinates for src
                    u, v = i-y_offset, j-x_offset
                    src_j, src_i, scale = h_inv@np.array([v, u, 1])
                    src_i, src_j = int(src_i/scale), int(src_j/scale)

                    # Checking if cordinates lie within the image
                    if(0 <= src_i < src_h and 0 <= src_j < src_w):
                        imgout[i, j] = src[src_i, src_j]

    else:
        imgout[y_offset:y_offset+src_h, x_offset:x_offset+src_w] = src
        #imgout[y_offset-src_h//2:y_offset+src_h//2, x_offset-src_w//2:x_offset+src_w//2] = src
    # Creating a alpha mask of the transformed image
    mask = np.sum(imgout, axis=2).astype(bool)
    return imgout, mask

def blend(images, masks, n=5):

    assert(images[0].shape[0] % pow(2, n) ==
           0 and images[0].shape[1] % pow(2, n) == 0)

    # Defining dictionaries for various pyramids
    g_pyramids = {}
    l_pyramids = {}

    H, W, C = images[0].shape

    # Calculating pyramids for various images before hand
    for i in range(len(images)):

        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G]
        for _ in range(n):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(G)

        # Laplacian Pyramids
        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i])-2, -1, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = cv2.subtract(G, G_up)
            l_pyramids[i].append(L)

    # Blending Pyramids
    common_mask = masks[0].copy()
    common_image = images[0].copy()
    common_pyramids = [l_pyramids[0][i].copy()
                       for i in range(len(l_pyramids[0]))]

    ls_ = None
    # We take one image, blend it with our final image, and then repeat for
    # n images
    for i in range(1, len(images)):

        # To decide which is left/right
        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        if np.max(x1) > np.max(x2):
            left_py = l_pyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = l_pyramids[i]

        # To check if the two pictures need to be blended are overlapping or not
        mask_intersection = np.bitwise_and(common_mask, masks[i])

        if True in mask_intersection:
            # If images blend, we need to find the center of the overlap
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)

            # We get the split point
            split = ((x_max-x_min)/2 + x_min)/W

            # Finally we add the pyramids
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = np.hstack(
                    (la[:, 0:int(split*cols)], lb[:, int(split*cols):]))
                LS.append(ls)

        else:
            print("hey there")
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = la + lb
                LS.append(ls)

        # Reconstructing the image
        ls_ = LS[0]
        for j in range(1, n+1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[j])

        # Preparing the commong image for next image to be added
        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return ls_

def blend_feathering(images, masks):

    

    H, W, C = images[0].shape


    common_mask = masks[0].copy()
    common_image = images[0].copy()
    #common_image = cv2.GaussianBlur(common_image,(3,3),0)

    ls_ = None
    # We take one image, blend it with our final image, and then repeat for
    # n images
    for i in range(1, len(images)):

        # To decide which is left/right
        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)
        
        blending_image = images[i].copy()
        #blending_image = cv2.GaussianBlur(blending_image,(3,3),0)

        if np.max(x1) > np.max(x2):
            left_py = blending_image
            right_py = common_image

        else:
            left_py = common_image
            right_py = blending_image

        # To check if the two pictures need to be blended are overlapping or not
        mask_intersection = np.bitwise_and(common_mask, masks[i])

        if True in mask_intersection:
            # If images blend, we need to find the center of the overlap
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)

            # We get the split point
            split = ((x_max-x_min)/2 + x_min)/W

            # Finally we add the pyramids

            rows, cols, dpt = left_py.shape
            ls = np.hstack((left_py[:, 0:int(split*cols)], right_py[:, int(split*cols):]))
            ls = cv2.GaussianBlur(ls,(3,3),0)

        else:

            rows, cols, dpt = left_py.shape
            ls = left_py + right_py



        # Preparing the commong image for next image to be added
        common_image = ls
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)

    return ls


image_directory_path = sys.argv[1]
# ex_no = input("which image 1,2,3,4?:")
# image_directory_path = "../data/{}".format(ex_no)

transformation = "affine"                          
blend_method = "pyramid"


image_names = os.listdir(image_directory_path)
image_names.sort()
n = len(image_names)    
print("No. of images to mosaic:",n)
increase_panoroma_height_by = 2 



Images = readImages(image_directory_path)


H, W, C = np.array(Images[0].shape)*[increase_panoroma_height_by, len(image_names), 1]     # Finding shape of final image



# Image Template for final image
img_f = np.zeros((H, W, C))
img_outputs = []
masks = []

print(f"\n||Setting the base image as {n//2}.||")
img, mask = warp(Images[n//2], None, img_f.copy(), H//2, W//2)

img_outputs.append(img)
masks.append(mask)
left_H = []
right_H = []

for i in range(1, len(Images)//2+1):

    # right
    if (n//2+i < n) and ((n//2+i-1) < n):  
        print(f"\n||For image {n//2+i}||")
        print("Caculating POC(s)")
        matrix = featureMatching(Images[n//2+i], Images[n//2+(i-1)],transformation)
        print("Performing RANSAC")
        right_H.append(matrix)
        print("Warping Image")
        img, mask = warp(Images[n//2+i], right_H[::-1],img_f.copy(), H//2, W//2)
        img_outputs.append(img)
        masks.append(mask)
        
    if (n//2-i >= 0) and ((n//2+i-1) >= 0):
        print(f"\n||For image {n//2-i}||")
        print("Caculating POC(s)----")
        print(n//2-i)
        matrix = featureMatching(Images[n//2-i], Images[n//2-(i-1)],transformation)
        
        print("Performing RANSAC")
        left_H.append(matrix)
        print(left_H[-1])
        print("Warping Image")
        img, mask = warp(Images[n//2-i], left_H[::-1],img_f.copy(), H//2, W//2)
        img_outputs.append(img)
        masks.append(mask)
        
# Blending all the images together
print("Please wait, Image Blending...")

if blend_method=="feathering":
    uncropped = blend_feathering(img_outputs, masks)
elif blend_method=="pyramid" :
    uncropped = blend(img_outputs, masks)

print("Image Blended, Final Cropping")
# Creating a mask of the panaroma
mask = np.sum(uncropped, axis=2).astype(bool)

# Finding appropriate bounding box
yy, xx = np.where(mask == 1)
x_min, x_max = np.min(xx), np.max(xx)
y_min, y_max = np.min(yy), np.max(yy)

# Croping and saving
final = uncropped[y_min:y_max, x_min:x_max]
cv2.imwrite("Panaroma_Image_{}_{}.jpg".format(transformation, blend_method), final)
print("Succesfully Saved image as Panaroma_Image.jpg.")

