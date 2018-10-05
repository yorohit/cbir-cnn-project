import shutil as sh
import numpy as np
import os

directory = '/home/rohit/pro/Rotated/GHIM-10K/'
dest = '/home/rohit/pro/Data/GHIM-10K/90-10/'
total = 1000

train_start = 0
num_class = 20

train_index = 0
test_index = 0

start = 0
end = 450

for i in range(1, num_class+1):

	if not os.path.exists(dest + 'train/' + str(i)):
		os.makedirs(dest + 'train/' + str(i))
	
	# if not os.path.exists(dest + 'val/' + str(i)):
	# 	os.makedirs(dest + 'val/' + str(i))

	for j in range(start, end):

		j = str(j).zfill(4)
		sh.copy(directory + j + '.jpg', dest + 'train/' + str(i) + '/' + j + '.jpg')
		# sh.copy(directory + j + '.jpg', dest + 'val/' + str(i) + '/' + j + '.jpg')

	start = start + 500
	end = end + 500	
