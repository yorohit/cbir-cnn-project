import os

path = '/home/rohit/pro/train/4/'

for file in sorted(os.listdir(path)):
	# print file
	os.rename(path + file, path + file[0:5	] + 'png')
