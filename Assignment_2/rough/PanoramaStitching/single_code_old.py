import os
import cv2
import numpy as np

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
        
    return None

def warp(src, homography, imgout, y_offset, x_offset):
    """
    This function warps the image according to the homography matrix and places the warped image
    at (y_offset,x_offset) in imgout and returns the final image.

    src: input image that needs to be warped
    homography: array of homography matrix required to bring src to base axes
    imgout: image to which src is to be transformed
    y_offset,x_offset: offsets of base image
    """

    # Getting the shapes
    H, W, C = imgout.shape
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
    """
    Image blending using Image Pyramids. We calculate Gaussian Pyramids using OpenCV.add()
    Once we have the Gaussian Pyramids, we take their differences to find Laplacian Pyramids
    or DOG(Difference of Gaussians). Then we add all the Laplacian Pyramids according to the
    seam/edge of the overlapping image. Finally we upscale all the Laplasian Pyramids to
    reconstruct the final image.

    images: array of all the images to be blended
    masks: array of corresponding alpha mask of the images
    n: max level of pyramids to be calculated.
    [NOTE: that image size should be a multiple of 2**n.]
    """

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

def poisson_blending(img_outputs, masks):
    panorma = img_outputs[0]
    #cv2.imshow("panoroma",panorma)
    cv2.imwrite("panoroma.jpg",panorma)
    

    #current_mask = masks[0]
    for i in range(1,len(img_outputs)):

        #Read images : src image will be cloned into dst
        obj= img_outputs[i]
        #cv2.imshow("obj",obj)
        cv2.imwrite("obj.jpg",obj)
        mask = masks[i]
        img_mask = np.array(mask) * 255

        
        print("\nPoisson Blending: \n")
        
        # print("mask dtype =",mask.dtype())
        # print("img_mask dtype =",img_mask.dtype())
        # print("panorma dtype = ",panorma.dtype())

        cv2.imwrite("mymask.jpg",img_mask)
        img_mask = cv2.imread("mymask.jpg",cv2.IMREAD_GRAYSCALE)
        #cv2.imshow("mask",img_mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # The location of the center of the object in the panorma
        y, x = np.where(img_mask==255)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        center = ((x_min+x_max)//2, (y_min+y_max)//2 ) 

        dummy_image = np.zeros(obj.shape)
        dummy_image = cv2.circle(dummy_image, center, radius=20, color=(0, 0, 255), thickness=10)
        cv2.imwrite("poisson_output/dummy_image.jpg",dummy_image)


        cv2.imwrite("poisson_output/object.jpg",obj)
        cv2.imwrite("poisson_output/panorma.jpg", panorma)
        cv2.imwrite("poisson_output/img_mask.jpg",img_mask)
        print("center coordinate:",center)



        print("obj shape =",obj.shape)
        print("img_mask shape =",img_mask.shape)
        print("panorma shape = ",panorma.shape)
        print("mask type =",type(mask))
        print("img_mask type =",type(img_mask))
        print("panorma type = ",type(panorma))
        # Seamlessly clone src into dst and put the results in output
        panorma = cv2.seamlessClone(obj, panorma, img_mask, center, cv2.NORMAL_CLONE)
        #mixed_clone = cv2.seamlessClone(obj, panorma, img_mask, center, cv2.MIXED_CLONE)
        cv2.imwrite("poisson_output/{}.jpg".format(i),panorma)
        # Write results
        #cv2.imwrite("opencv-normal-clone-example.jpg", panorma)
        #cv2.imwrite("opencv-mixed-clone-example.jpg", mixed_clone)


ex_no = input("which image 1,2,3,4?:")
image_directory_path = "../data/{}".format(ex_no)

image_names = os.listdir(image_directory_path)
image_names.sort()
#imggg = cv2.imread(os.path.join(image_directory_path, image_names[0]))
# cv2.imshow("first image",imggg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

n = len(image_names)    
print("no. of images:",n)
increase_panoroma_height_by = 2 

transformation = "affine"                          

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
        print(right_H[-1])
        print("Warping Image")
        img, mask = warp(Images[n//2+i], right_H[::-1],img_f.copy(), H//2, W//2)
        img_outputs.append(img)
        cv2.imwrite("outputs/{}.jpg".format(n//2+i), img)
        # print(mask.shape)
        # print(type(mask))
        # print(type(mask[0,0]))
        # print(img.shape)
        # print(type(img))
        # print(type(img[0,0]))
        img_mask = np.array(mask) * 255
        cv2.imwrite("outputs/mask_{}.jpg".format(n//2+i), img_mask)
        masks.append(mask)
        
    if (n//2-i >= 0) and ((n//2-i+1) >= 0):
        print(f"\n||For image {n//2-i}||")
        print("Caculating POC(s)----")
        print(n//2-i)
        matrix = featureMatching(Images[n//2-i], Images[n//2-(i-1)])
        
        print("Performing RANSAC")
        left_H.append(matrix)
        print(left_H[-1])
        print("Warping Image")
        img, mask = warp(Images[n//2-i], left_H[::-1],img_f.copy(), H//2, W//2)
        img_outputs.append(img)
        cv2.imwrite("outputs/{}.jpg".format(n//2-i), img)
        
        img_mask = np.array(mask) * 255
        cv2.imwrite("outputs/mask_{}.jpg".format(n//2-i), img_mask)
    
        masks.append(mask)
    

#unblended image
# cv2.imwrite("unblended_{}_{}.jpg".format(ex_no,transformation), img_outputs[-1])

# Blending all the images together
print("Please wait, Image Blending...")
poisson_blending(img_outputs, masks)
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
cv2.imwrite("Panaroma_Image_{}_{}.jpg".format(ex_no, transformation), final)
print("Succesfully Saved image as Panaroma_Image.jpg.")