import shutil as sh
import numpy as np
import os

directory = '/home/rohit/Image Datasets/C1K/'
total = 1000

train_start = 0
num_class = 10

train_index = 0
test_index = 0

start = 70
end = 100

for i in range(1, num_class+1):

	if not os.path.exists('/home/rohit/Image Datasets/C1KRotated/train/' + str(i)):
		os.makedirs('/home/rohit/Image Datasets/C1KRotated/train/' + str(i))
	
	if not os.path.exists('/home/rohit/Image Datasets/C1KRotated/val/' + str(i)):
	 	os.makedirs('/home/rohit/Image Datasets/C1KRotated/val/' + str(i))

	for j in range(start, end):

		j = str(j).zfill(3)
		# sh.copy(directory + j + '.jpg', '/home/rohit/Image Datasets/C1KRotated/train/' + str(i) + '/' + j + '.jpg')
		sh.copy(directory + j + '.jpg', '/home/rohit/Image Datasets/C1KRotated/val/' + str(i) + '/' + j + '.jpg')

	start = start + 100
	end = end + 100	
