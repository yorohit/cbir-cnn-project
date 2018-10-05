import numpy as np
import cPickle
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import time

# Cluster for each class in training set
def class_cluster(x_train):

	kmeans_model = KMeans(n_clusters = 3, random_state = 42)
	# clusters = np.zeros(num_class, num_clusters, num_dim)
	# start = 0
	kmeans = []

	for i in xrange(1, 21):
		
		x_current = x_train[i-1].reshape(350, -1)
		# print x_current.shape
		# start += 500

		temp = kmeans_model.fit(x_current)
		kmeans.append(temp)
		
		# kmeans.cluster_centers_
		# kmeans.predict

	return kmeans


def class_predict(x_test, model):

	return model.predict(x_test)

def testing(x_test, x_train, weights_path, cluster_output):

	print "Enter Testing"
	model = load_trained_model(weights_path)
	x_test_block = x_test
	x_test = x_test.reshape(x_test.shape[0], -1)
	num_test = x_test.shape[0]
	
	print len(cluster_output)
	
	for i in xrange(num_test):
		if i % 150 == 0:
			print "Class Over"

		label = np.argmax(class_predict(x_test_block[np.newaxis, i], model))
		print label,
		class_kmeans = cluster_output[label]
		centroids = class_kmeans.cluster_centers_
		# print centroids

		dist = dict()
		for j in xrange(centroids.shape[0]):
			temp = np.sqrt(np.sum((centroids[j] - x_test[i])**2))
			dist[temp] = j

		closest_centroid = dist.items()[0][1]
		# print closest_centroid

		# Write this
		current = x_train[label].reshape(350, -1)
		# print current.shape
		# print class_kmeans.predict(current)
		closest_cluster = current[np.argwhere(class_kmeans.predict(current) == closest_centroid)]
		# print closest_cluster.shape

		dist = dict()
		for j in xrange(closest_cluster.shape[0]):
			temp = np.sqrt(np.sum((closest_cluster[j] - x_test[i])**2))
			dist[temp] = j

		ctr = 0
		top5 = []
		for k, v in dist.items():
			if ctr < 5:
				top5.append(v)
				ctr += 1
			else:
				break

		print top5


def get_top5(test_sample, dataset, split, components):


	# train = np.load(open('70-30/features/bottleneck_features_train.npy'))
	print "Started"
	x_test = np.load(open('/home/rohit/pro/70-30/features/bottleneck_features_validation.npy'))
	print "Loaded x_test"
	K = 3 # Global
	weights_path = 'bottleneck_fc_model.h5'

	print "Loading train"
	x_train = []
	for i in xrange(1, 21):
		x_train.append(np.load(open('features/train/train' + str(i) +'.npy')))
	print "Train Done"

	with open('cluster_models', 'rb') as f:
		cluster_output = cPickle.load(f)

	print "Started Clustering"
	start = time.time()
	# cluster_output = class_cluster(x_train)
	print time.time() - start

	# with open('cluster_models', 'wb') as f:
	# 	cPickle.dump(cluster_output, f)

	print "Clustering Done"
	print "Started Testing"
	testing(x_test, x_train, weights_path, cluster_output)
	print "Done"

