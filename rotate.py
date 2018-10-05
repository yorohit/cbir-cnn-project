import numpy as np


import os, cv2

srcPath = '/home/rohit/Image Datasets/Original/GHIM-10K/'
destPath = '/home/rohit/pro/Rotated/GHIM-10K/'

for file in sorted(os.listdir(srcPath)):

	img = cv2.imread(srcPath + file)
	
	if img.shape[0] == 300:
		imgRot = np.transpose(img, (1, 0, 2))
	else:
		imgRot = img
	
	cv2.imwrite(destPath + file, imgRot)