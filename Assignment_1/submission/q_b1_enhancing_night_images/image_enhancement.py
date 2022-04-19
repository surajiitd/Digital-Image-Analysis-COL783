import numpy as np
import argparse
import cv2
import os

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_lookup_table(gamma):
    table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table

def perform_gamma_correction(image, gamma):
	table = generate_lookup_table(gamma)
	return cv2.LUT(image, table)

# Command : python3 image_enhancement.py -i "../data/night" -o "output"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', type=str, help='path to input directory containing dir 1,2,3')
    parser.add_argument('-o','--out', type=str, help='path to store output')
    args = parser.parse_args()

    image_id = int(input("enter image id 1,2,3 ?: "))

    outputFolderPath = os.path.join(args.out,"{}".format(image_id))
    inputImage = cv2.imread(os.path.join(args.inp, "{}/night_mode_off.jpg".format(image_id)) )
    expectedImage = cv2.imread(os.path.join(args.inp, "{}/night_mode_on.jpg".format(image_id)) )
    make_dir(path=outputFolderPath)

    imgToyuv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YUV)
    imgTohsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
    imgTolab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
    imgToycrcb = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YCrCb)


    #applying gamma correction
    gamma_correction_output_path = os.path.join(outputFolderPath,"Output_gamma_correction_"+str(image_id))
    make_dir(path=gamma_correction_output_path)

    for gamma in np.arange(0.5,3.5,0.5):
        gamma_adjusted = perform_gamma_correction(inputImage, gamma)
        cv2.putText(gamma_adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(gamma_correction_output_path,"gamma_corrected_output_g_"+str(gamma)+".jpg"), gamma_adjusted)


    #applying histogram equilization

    histogram_equilization_output_path = os.path.join(outputFolderPath,"Output_histogram_equilization_"+str(image_id))
    make_dir(path=histogram_equilization_output_path)

    #YUV
    imgToyuv[:,:,0] = cv2.equalizeHist(imgToyuv[:,:,0])
    imgOutputyuv = cv2.cvtColor(imgToyuv, cv2.COLOR_YUV2BGR)
    cv2.putText(imgOutputyuv, "YUV", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(histogram_equilization_output_path,"histogram_equilization_output_yuv.jpg"), imgOutputyuv)

    #HSV
    imgTohsv[:,:,2] = cv2.equalizeHist(imgTohsv[:,:,2])
    imgOutputhsv = cv2.cvtColor(imgTohsv, cv2.COLOR_HSV2BGR)
    cv2.putText(imgOutputhsv, "HSV", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(histogram_equilization_output_path,"histogram_equilization_output_hsv.jpg"), imgOutputhsv)

    #LAB
    imgTolab[:,:,0] = cv2.equalizeHist(imgTolab[:,:,0])
    imgOutputlab = cv2.cvtColor(imgTolab, cv2.COLOR_LAB2BGR)
    cv2.putText(imgOutputlab, "LAB", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(histogram_equilization_output_path,"histogram_equilization_output_lab.jpg"), imgOutputlab)

    #YCRCB
    imgToycrcb[:,:,0] = cv2.equalizeHist(imgToycrcb[:,:,0])
    imgOutputycrcb = cv2.cvtColor(imgToycrcb, cv2.COLOR_YCrCb2BGR)
    cv2.putText(imgOutputycrcb, "YCRCB", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(histogram_equilization_output_path,"histogram_equilization_output_ycrcb.jpg"), imgOutputycrcb)

    #applying adaptive histogram equilazation.

    adaptive_histogram_equilization_output_path = os.path.join(outputFolderPath,"Output_adaptive_histogram_equilization_"+str(image_id))
    make_dir(path=adaptive_histogram_equilization_output_path)


    for i in range(2,6):
        for j in range(1,30,2):
            print(i,j)
            cliplimit = i
            tileGridSize = j
            
            #inputImage = cv2.imread(inputImagePath+"foggy.png")
            #expectedImage = cv2.imread(inputImagePath+"clear.png")
            imgToyuv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YUV)
            imgTohsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
            imgTolab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
            imgToycrcb = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YCrCb)
            
            clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(tileGridSize, tileGridSize))
            
            #YUV
            imgToyuv[:,:,0] = clahe.apply(imgToyuv[:,:,0])
            imgOutputyuv = cv2.cvtColor(imgToyuv, cv2.COLOR_YUV2BGR)
            cv2.putText(imgOutputyuv, "YUV"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_yuv"+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputyuv)
            
            #HSV
            imgTohsv[:,:,2] = clahe.apply(imgTohsv[:,:,2])
            imgOutputhsv = cv2.cvtColor(imgTohsv, cv2.COLOR_HSV2BGR)
            cv2.putText(imgOutputhsv, "HSV"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_hsv"+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputhsv)
            
            #LAB
            imgTolab[:,:,0] = clahe.apply(imgTolab[:,:,0])
            imgOutputlab = cv2.cvtColor(imgTolab, cv2.COLOR_LAB2BGR)
            cv2.putText(imgOutputlab, "LAB"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_lab"+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputlab)
            
            #YCRCB
            imgToycrcb[:,:,0] = clahe.apply(imgToycrcb[:,:,0])
            imgOutputycrcb = cv2.cvtColor(imgToycrcb, cv2.COLOR_YCrCb2BGR)
            cv2.putText(imgOutputycrcb, "YCRCB"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_ycrcb"+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputycrcb)
            
    #applying gamma correction + adaptive histogram equilazation.

    gc_adaptive_histogram_equilization_output_path = os.path.join(outputFolderPath,"Output_gc+adaptive_histogram_equilization_"+str(image_id))
    make_dir(path=gc_adaptive_histogram_equilization_output_path)

    for gamma in np.arange(0.5,3.5,0.5):
        for i in range(2,6):
            for j in range(1,30,2):
                print(i,j)
                cliplimit = i
                tileGridSize = j
                
                #inputImage = cv2.imread(inputImagePath+"foggy.png")
                gamma_adjusted = perform_gamma_correction(inputImage, gamma)
                inputImage = gamma_adjusted
                imgToyuv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YUV)
                imgTohsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
                imgTolab = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
                imgToycrcb = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YCrCb)
                
                clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(tileGridSize, tileGridSize))
                
                #YUV
                imgToyuv[:,:,0] = clahe.apply(imgToyuv[:,:,0])
                imgOutputyuv = cv2.cvtColor(imgToyuv, cv2.COLOR_YUV2BGR)
                cv2.putText(imgOutputyuv, "g: "+str(gamma)+"YUV"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imwrite(os.path.join(gc_adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_yuv"+"g_"+str(gamma)+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputyuv)
                
                #HSV
                imgTohsv[:,:,2] = clahe.apply(imgTohsv[:,:,2])
                imgOutputhsv = cv2.cvtColor(imgTohsv, cv2.COLOR_HSV2BGR)
                cv2.putText(imgOutputhsv, "g: "+str(gamma)+"HSV"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imwrite(os.path.join(gc_adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_hsv"+"g_"+str(gamma)+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputhsv)
                
                #LAB
                imgTolab[:,:,0] = clahe.apply(imgTolab[:,:,0])
                imgOutputlab = cv2.cvtColor(imgTolab, cv2.COLOR_LAB2BGR)
                cv2.putText(imgOutputlab, "g: "+str(gamma)+"LAB"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imwrite(os.path.join(gc_adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_lab"+"g_"+str(gamma)+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputlab)
                
                #YCRCB
                imgToycrcb[:,:,0] = clahe.apply(imgToycrcb[:,:,0])
                imgOutputycrcb = cv2.cvtColor(imgToycrcb, cv2.COLOR_YCrCb2BGR)
                cv2.putText(imgOutputycrcb, "g: "+str(gamma)+"YCRCB"+" tile:"+str(j)+" clip:"+str(i), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imwrite(os.path.join(gc_adaptive_histogram_equilization_output_path,"adaptive_histogram_equilization_output_ycrcb"+"g_"+str(gamma)+"tile_"+str(j)+" clip_"+str(i)+".jpg"), imgOutputycrcb)
            

