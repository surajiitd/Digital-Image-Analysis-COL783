import cv2
import numpy as np
import glob
import os
import argparse
from scipy.interpolate import UnivariateSpline
from utils.Airlight import Airlight
from utils.BoundCon import BoundCon
from utils.CalTransmission import CalTransmission
from utils.removeHaze import removeHaze
from utils.exposure_enhancement import enhance_image_exposure


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def FrameCapture(path,output_path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
  
    while success:
        success, image = vidObj.read()
        if not success:
            break
        cv2.imwrite(os.path.join(output_path,"%d.jpg" % count), image)
  
        count += 1

#Command : python3 video_enhancement.py -i "../data/night_time_video_iitd.mp4" -o "./"
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inp', type=str, help='path to input video', default="../data/night_time_video_iitd.mp4")
parser.add_argument('-o','--out', type=str, help='path to output video', default="./")
args = parser.parse_args() 

inputVideoPath = args.inp  #"../data/night_time_video_iitd.mp4"
outputVideoPath = args.out  #"./"


outputFramesPath = os.path.join(outputVideoPath,"video_to_frames")
make_dir(path=outputFramesPath)

FrameCapture(inputVideoPath, outputFramesPath)

img_array = []              
for filename in range(375):
    filename = str(filename)+".jpg"
    print(filename)
    HazeImg = cv2.imread(os.path.join(outputFramesPath,filename))
    
    
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A,0.5)
    
    HazeCorrectedImg  = cv2.resize(HazeCorrectedImg, (640,480),interpolation = cv2.INTER_AREA)
    height, width, layers = HazeCorrectedImg.shape
    size = (width,height)
    
    g = 0.6
    l = 0
    enhanced_image = enhance_image_exposure(HazeCorrectedImg, g, l,True,sigma= 3, bc= 1 , bs= 1, be= 1 , eps= 1e-3)
    
    
    img_array.append(enhanced_image)
 
 
out = cv2.VideoWriter(os.path.join(outputVideoPath,'enhanced_final.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
