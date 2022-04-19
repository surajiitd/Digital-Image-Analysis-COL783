# Assignment 1

## Commands to run :

## A. Demosaiking
 - go to q_a_demosaicing and execute:
	```
	python3 q1.py -i "../data/demosaicing"
	```


## B. Image enhancement

#### 1. Night-time images:
 - go to q_b1_enhancing_night_images and execute:
 - To execute conventional methods of Image Processing:
	```
	python3 image_enhancement.py -i "../data/night" -o "output" 
	```


 - To execute method implemented in paper that we referring:
	```
	python3 demo.py 
	```


#### 2. Foggy images:
 - go to q_b2_enhancing_foggy_images and execute:
	``` 
	python3 fog.py -i "../data/fog" -o "output" 
	```


## C. Video enhancement
 - go to q_c_video_enhancement and execute:
	``` 
	python3 video_enhancement.py -i "../data/night_time_video_iitd.mp4" -o "./" 
	```
	

## References

<a id="1">[1]</a> 
Qing Zhang and Yongwei Nie and Weishi Zheng.
"Dual Illumination Estimation for Robust Exposure Correction".
Computer Graphics Forum, 2019, 38.

<a id="1">[2]</a> 
G. Meng, Y. Wang, J. Duan, S. Xiang and C. Pan, "Efficient Image Dehazing with Boundary Constraint and Contextual Regularization," 2013 IEEE International Conference on Computer Vision, 2013, pp. 617-624, doi: 10.1109/ICCV.2013.82.

<a id="1">[3]</a> 
https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python

<a id="1">[4]</a> 
https://github.com/pvnieo/Low-light-Image-Enhancement