

import numpy as np
import cv2

cap = cv2.VideoCapture("./data/night_time_video_iitd.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc,video_fps, frame_size) 

#constants in VideoWriter
#Ref: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5


frame_size = (int(cap.get(CAP_PROP_FRAME_WIDTH)),int(cap.get(CAP_PROP_FRAME_HEIGHT)))
video_fps = cap.get(CAP_PROP_FPS)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

    	# DO THE OPERATIONS ON FRAME or STORE FRAMES IN ARRAY, THEN LATER DO out.write()
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()