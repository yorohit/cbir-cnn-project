from __future__ import print_function

import numpy as np

import cPickle, time, os

from sklearn.cluster import KMeans
from sklearn import preprocessing


path = "/home/rohit/pro/latest/feature_split_fol/"

def load_data(folder):
	
	x = np.load(open(folder))
	return x

def class_cluster(x_train):
	
	kmeans_model = KMeans(n_clusters = 3, random_state = 42)
	temp = kmeans_model.fit(x_train)
	return temp

def testing(path, test_sample, split, x_train):
	
	# y = load_data("/home/rohit/pro/Features/Pascal/" + split + "/pascal_labels_test.npy")
	y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(int)

	num_test = 1

	for i in xrange(num_test):
		
		top5 = []
		dist1 = dict()
		dist = dict()

		start = time.clock()
		label = 0
		# for label, vl in enumerate(y[i]):
			
			# if(vl == 0):
			# 	continue
			
			#class_kmeans = cluster_output[label]
			#centroids = class_kmeans.cluster_centers_

			
			# for j in xrange(centroids.shape[0]):
				
			# 	temp = np.sqrt(np.sum((centroids[j] - test_sample)**2))
			# 	dist[temp] = j

			# closest_centroid = dist.items()[0][1]
			# # print closest_centroid

			# #with open(path+"/"+str(label)+"/"+'data', "rb") as input_file:
			# #	current = cPickle.load(input_file)
			# current = x_train[label]

			# current = np.array(current)

			# closest_cluster = current[(np.argwhere(class_kmeans.predict(current) == closest_centroid)).astype(int)]
			# closest_cluster=[]
			# for i_j in current:
			# 	i_j = np.array(i_j)
			# 	# i_j = np.reshape(1, -1)
			# 	if(class_kmeans.predict(i_j) == closest_centroid):
			# 		closest_cluster.append(i_j)

			# print(time.clock() - start)

		print("X: %s" % x_train[label].shape[0])
		
		# s = time.clock()
		for j in xrange(x_train[label].shape[0]):
		# for j in xrange(1000):
			temp = np.sqrt(np.sum((x_train[label][j] - test_sample)**2))
			dist1[temp] = j + 0.01*label

			# print("jddj: %s" % (time.clock() - s))

		ctr = 0
			
		for k, v in dist1.items():
			if ctr < 5:
				top5.append(v)
				ctr += 1
			else:
				break

		print(top5)	

		print(time.clock() - start)


def get_top5(test_sample, dataset, split, components):
	
	class_models = []
	x_train = []
	# print("Yo")
	for i in range(0, 2):
		
		with open('/home/rohit/pro/pascal_images_split/60-40/train_label_split/' + str(i) + '/data', "rb") as input_file:
			# print(path + dataset + "/" + split + "/" + str(components) + "/" + str(i) + "/" + 'data')
			# path + dataset + "/" + split + "/original/" + str(i) + "/" + 'data'
			x = cPickle.load(input_file)
			print(len(x))
			x = np.array(x)
			x = x.reshape(x.shape[0], -1)
			# x = preprocessing.scale(x)
		# ttt = path + dataset + "/" + split + "/" + str(components) + "/" + str(i) + "/" + 'data'

		#temp = class_cluster(x)
		x_train.append(x)

		#class_models.append(temp)
	
	testing(path, test_sample, split, x_train)


