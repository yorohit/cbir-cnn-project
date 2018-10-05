from __future__ import print_function

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, regularizers
from keras.callbacks import EarlyStopping

import time, cPickle, timeit, h5py

import clustering_pascal, rest_cluster, brute_force_rest, brute_force_pascal

dataset = "C1K"

split = "60-40"
num_samples_per_class = 500
num_class = 20

components = 50	

# num_train_per_class = (split[0:2].astype(int)) * 0.01 * num_samples_per_class
num_train_per_class = int(int(split[0:2]) * 0.01 * num_samples_per_class)
# print(num_train_per_class)
# components_list = [50, 75, 100, 125, 150, 200, 300, 400]

def get_input():

	# with open(path + "x_test_red", "rb") as f:
	# 	x_test_red = cPickle.load(f)

	# x_test_red = np.load(path)

	with open('/home/rohit/pro/pascal_images_split/60-40/val_label_split/0/data') as f:
		x_test_red = cPickle.load(f)

	return x_test_red[0]

# for components in components_list:

# path = "../Features/" + dataset + "/" + split + "/PCA/" + str(components) + "/"
# model_path = "Models/" + dataset + "/" + split + "/" + str(components)
# path = "../Features/" + dataset + "/" + split + "/ghim_cnn_test.npy"
# model_path = "Models/" + dataset + "/" + split + "/original"

# test_sample = get_input().reshape(-1, components)
test_sample = get_input()[np.newaxis, :]

# model = load_model(model_path)

# start = time.time()
# print(model.predict(test_sample))
# print(time.time() - start)

print(components)
test_sample = test_sample.reshape(test_sample.shape[0], -1)
clustering_pascal.get_top5(test_sample, dataset, split, components)
# rest_cluster.get_top5(test_sample, dataset, split, components, num_class, num_train_per_class)
# brute_force_pascal.get_top5(test_sample, dataset, split, components)
# brute_force_rest.get_top5(test_sample, dataset, split, components, num_class, num_train_per_class)
# rest_cluster.get_top5(test_sample, dataset, split, components, num_class, num_train_per_class)
