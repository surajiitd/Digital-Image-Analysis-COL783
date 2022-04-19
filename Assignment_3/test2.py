	
	images = ["inp1_forest.png"]


	for i in range(len(images)):
		img = cv2.imread(  )
		
		# Apply different transformations to Low Resolution image.
		flip0 = cv2.imread(  )
		cv2.imwrite(images[i][:-4]+"_flip0_{}_enhanced_p.png".format(distance_metric),flip0)
		flip1 = cv2.imread(  )
		cv2.imwrite(images[i][:-4]+"_flip1_{}_enhanced_p.png".format(distance_metric),flip1)
		rotated90 = cv2.imread(  )
		cv2.imwrite(images[i][:-4]+"_rotated90_{}_enhanced_p.png".format(distance_metric),rotated90)
		rotated180 = cv2.imread(  )
		cv2.imwrite(images[i][:-4]+"_rotated180_{}_enhanced_p.png".format(distance_metric),rotated180)
		
		# Call the SR function for all transformed images.
		flip0 = main_SR(flip0)
		flip1 = main_SR(flip1)
		rotated90 = main_SR(rotated90)
		rotated180 = main_SR(rotated180)

		# Apply the inverse Transformation to the High Resolution images.
		flip0_back = cv2.flip(flip0,0)
		flip1_back = cv2.flip(flip1,1)
		rotated90_back = cv2.rotate(rotated90, cv2.ROTATE_90_COUNTERCLOCKWISE)
		rotated180_back = cv2.rotate(rotated180, cv2.ROTATE_180)	
		
		flip0_back = flip0_back.astype('int')
		flip1_back = flip1_back.astype('int')
		rotated90_back = rotated90_back.astype('int')
		rotated180_back = rotated180_back.astype('int')

		# Take the average of all output images.
		avg_img = (flip0_back + flip1_back + rotated90_back + rotated180_back)//4
		avg_img = avg_img.astype('uint8')

		cv2.imwrite(images[i][:-4]+"_SR_paper_{}_enhanced_p.png".format(distance_metric),avg_img)