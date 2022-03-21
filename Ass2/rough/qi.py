import numpy as np
import imutils
import cv2

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
	# initialize the output visualization image
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB
	# loop over the matches
	for ((trainIdx, queryIdx), s) in zip(matches, status):
		# only process the match if the keypoint was successfully
		# matched
		if s == 1:
			# draw the match
			ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
			ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
			cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
	# return the visualization
	return vis

def detectAndDescribe(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, features) = descriptor.detectAndCompute(image, None)

	kps = np.float32([kp.pt for kp in kps])
	return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
	ratio, reprojThresh):
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	matches = []

	for m in rawMatches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))

	if len(matches) > 4:

		ptsA = np.float32([kpsA[i] for (_, i) in matches])
		ptsB = np.float32([kpsB[i] for (i, _) in matches])
		(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
			reprojThresh)

		return (matches, H, status)
	return None



def stitch(images, ratio=0.75, reprojThresh=4.0,showMatches=False):
		(imageB, imageA) = images
		(kpsA, featuresA) = detectAndDescribe(imageA)
		(kpsB, featuresB) = detectAndDescribe(imageB)
		M = matchKeypoints(kpsA, kpsB,featuresA, featuresB,ratio,reprojThresh)

		if M is None:
			return None

		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		if showMatches:
			vis = drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
			return (result, vis)
		return result

imageA = cv2.imread("data/1/1.jpg")
imageB = cv2.imread("data/1/2.jpg")
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

result = stitch([imageA, imageB], showMatches=False)
for i in range(3,7):
	new_image = cv2.imread("data/1/"+str(i)+".jpg")
	new_image = imutils.resize(new_image, width=400)
	result = imutils.resize(new_image, width=i*400)
	cv2.imshow("result",result)
	cv2.waitKey(0)
	result = stitch([result, new_image], showMatches=False)



cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
#cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)