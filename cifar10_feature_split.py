import os
import numpy as np
import pickle
from collections import defaultdict

from keras.datasets import cifar10

from sklearn.preprocessing import label_binarize

import read_data

path = "/home/rohit/pro/cifar10_split"

def load_data(folder):
	
	x = np.load(open(folder))
	return x

def create_folder(path, components = None):
	
	for i in range(0, 10):

		# if not os.path.exists(path + "/" + str(components) + "/" + str(i)):
		# 	os.makedirs(path + "/" + str(components) + "/" + str(i))

		if not os.path.exists(path + "/original/" + str(i)):
			os.makedirs(path + "/original/" + str(i))

def process(path, components = None):
	
	create_folder(path, components)
	
	# with open("/home/rohit/pro/Features/CIFAR10/" + "PCA/" + str(components) + "/x_train_red", "rb") as input_file:
	# 	x = pickle.load(input_file)


	x, y_train, _, _ = read_data.cifar10()
	# y_train = label_binarize(y_train, classes = list(np.arange(10)))

	# y = load_data("/home/rohit/pro/Features/" + "_labels_train.npy")

	storage_dict = defaultdict(list)
	
	for index, y_i in enumerate(y_train):
		for y_i_j,vl in enumerate(y_i):
			#print index,len(y_i)
			if(vl == 1):
				# storage_dict[path + "/" + str(components) + "/" + str(y_i_j)].append(x[index])
				storage_dict[path + "/original/" + str(y_i_j)].append(x[index])
				#print index,y_i_j


	#print storage_dict[0]


	for i in range(0, 10):
		
		# file_pi = open(path + "/" + str(components) + "/" + str(i) + "/data", 'w') 
		# pickle.dump(storage_dict[path + "/" + str(components) + "/" + str(i)], file_pi)

		file_pi = open(path + "/original/" + str(i) + "/data", 'w') 
		pickle.dump(storage_dict[path + "/original/" + str(i)], file_pi)		

		file_pi.close()

# split = "80-20"
# num_components = [50, 75, 100, 125, 150, 200, 300, 400]
# for components in num_components:
process(path)	
#create_folder(path)
#x=load_data("/home/rohit/pro/Features//90-10/_cnn_train.npy")











