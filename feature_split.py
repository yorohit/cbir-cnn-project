import os
import numpy as np
import cPickle
from collections import defaultdict


path = "/home/rohit/pro/latest/feature_split_fol/"

def load_data(folder):
	
	x = np.load(open(folder))
	return x

def create_folder(path, split):
	
	for i in range(0,20):

		if not os.path.exists(path + "Pascal/" + split + "/original/" + str(i)):
			os.makedirs(path + "Pascal/" + split + "/original/" + str(i))

def process(path, split):
	
	# create_folder(path, split, components)
	create_folder(path, split)

	# with open("/home/rohit/pro/Features/Pascal/" + split + "/PCA/" + str(components) + "/x_train_red", "rb") as input_file:
	# 	x = cPickle.load(input_file)

	input_file = "/home/rohit/pro/Features/Pascal/" + split + "/pascal_cnn_train.npy"
	x = np.load(input_file)

	y = load_data("/home/rohit/pro/Features/Pascal/" + split + "/pascal_labels_train.npy")

	storage_dict = defaultdict(list)
	
	for index,y_i in enumerate(y):
		for y_i_j,vl in enumerate(y_i):
			#print index,len(y_i)
			if(vl == 1):
				# storage_dict[path + "Pascal/" + split + "/" + str(components) + "/" + str(y_i_j)].append(x[index])
				storage_dict[path + "Pascal/" + split + "/original/" + str(y_i_j)].append(x[index])
				#print index,y_i_j


	#print storage_dict[0]


	for i in range(0,20):
		
		file_pi = open(path + "Pascal/" + split + "/original/" + str(i) + "/data", 'w') 
		cPickle.dump(storage_dict[path + "Pascal/" + split + "/original/" + str(i)], file_pi)
		file_pi.close()

split = "80-20"
# num_components = [50, 75, 100, 125, 150, 200, 300, 400]
# for components in num_components:
process(path, split)	
#create_folder(path)
#x=load_data("/home/rohit/pro/Features/Pascal/90-10/pascal_cnn_train.npy")











