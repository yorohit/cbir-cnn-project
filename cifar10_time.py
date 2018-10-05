from __future__ import print_function

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, regularizers
from keras.callbacks import EarlyStopping

import time, pickle, timeit, h5py

import clustering_pascal, rest_cluster, cifar10_clustering, read_data, brute_force_cifar10

dataset = "CIFAR10"

# split = "90-10"
# num_samples_per_class = 100
num_class = 10

components = 400

# num_train_per_class = (split[0:2].astype(int)) * 0.01 * num_samples_per_class
# num_train_per_class = int(int(split[0:2]) * 0.01 * num_samples_per_class)
# print(num_train_per_class)
components_list = [50, 75, 100, 125, 150, 200, 300, 400]


def get_input():

	# with open(path + "x_test_red", "rb") as f:
	# 	x_test_red = pickle.load(f)

	x_train, y_train, x_test_red, y_test = read_data.cifar10()

	# x_test_red = np.load(open(path))

	return x_test_red[0]

# for components in components_list:

# path = "../Features/" + dataset + "/PCA/" + str(components) + "/"
# model_path = "Models/" + dataset + "/" + str(components)
path = "../Features/" + dataset + "/PCA/" + str(components) + "/" + "/c1_cnn_test.npy"
model_path = "Models/" + dataset + "/original"

# test_sample = get_input().reshape(-1, components)
# test_sample = get_input()[np.newaxis, :]

model = load_model(model_path)

test_sample = get_input().reshape(1, -1)

# start = time.time()
# print(np.argmax(model.predict(test_sample)))
# print(time.time() - start)

test_sample = test_sample.reshape(test_sample.shape[0], -1)
# clustering_pascal.get_top5(test_sample, dataset, split, components)
# rest_cluster.get_top5(test_sample, dataset, split, components, num_class, num_train_per_class)
print(components)
#cifar10_clustering.get_top5(test_sample, components)
brute_force_cifar10.get_top5(test_sample, components)