# - Load a x_test_red file
# - Slice into classes and fit k means
# - call testing code


import numpy as np

import cPickle, time

from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import read_data

path = '/home/rohit/pro/Features/'

# dataset = "GHIM-10K"
# split = "60-40"
# components = 50

def class_cluster(x_train):
	
	kmeans_model = KMeans(n_clusters = 3, random_state = 42)
	temp = kmeans_model.fit(x_train)
	return temp


def get_clusters(path, dataset, split, components, num_class, num_train_per_class):

	cluster_models = []

	x_train = cPickle.load(open(path + dataset +  "/" + split + "/PCA/" + str(components) + "/x_train_red"))
	# x_train, _, _, _ = read_data.c1(split)
	print(x_train.shape)

	for i in range(num_class):

		x_split = x_train[i * num_train_per_class : (i+1) * num_train_per_class]
		print(x_split.shape)
		x_split = x_split.reshape(x_split.shape[0], -1)

		cluster_models.append(class_cluster(x_split))


	return cluster_models, x_train


def testing(test_sample, x_train, num_train_per_class, split):

	dist = dict()

	prediction = 0
	start = time.time()

	#class_kmeans = cluster_models[prediction]

	#centroids = class_kmeans.cluster_centers_

	# for j in xrange(x_train.shape[0]):
	# 	temp = np.sqrt(np.sum((x_train[j] - test_sample)**2))
	# 	dist[temp] = j

	# closest_centroid = dist.items()[0][1]

	# # current = x_train[prediction].reshape(num_train_per_class, -1)
	# current = x_train
	# current = current.reshape(current.shape[0], -1)

	# closest_cluster = current[np.argwhere(class_kmeans.predict(current) == closest_centroid)]

	# dist = dict()
	x_train = x_train[prediction : prediction + int(int(split[0:2]) * 0.01 * 500)]

	for j in xrange(x_train.shape[0]):
		temp = np.sqrt(np.sum((x_train[j] - test_sample)**2))
		dist[temp] = j

	ctr = 0
	top5 = []
	for k, v in dist.items():
		if ctr < 5:
			top5.append(v)
			ctr += 1
		else:
			break

	print(top5)
	print(time.time() - start)

def get_top5(test_sample, dataset, split, components, num_class, num_train_per_class):

	# cluster_models, x_train = get_clusters(path, dataset, split, components, num_class, num_train_per_class)
	# x_train, _, _, _ = read_data.c1(split)
	x_train = cPickle.load(open(path + dataset +  "/" + split + "/PCA/" + str(components) + "/x_train_red"))
	testing(test_sample, x_train, num_train_per_class, split)


