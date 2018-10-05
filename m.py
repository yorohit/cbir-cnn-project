import numpy as np

import read_data, classifiers, pca, MLP

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


import pickle, os

from xgboost import XGBClassifier

# Not required, mean and variance already close to zero
# x_train, x_test = read_data.preprocess(x_train, x_tes

def store_reduced_features(x_train, x_test, components, dataset, split = None):

	for num_components in components:

		# path = "../Features/" + dataset + "/" + split + "/PCA/" + str(num_components) + "/"
		path = "../Features/" + dataset + "/" + "/PCA/" + str(num_components) + "/"

		if not os.path.exists(path):
			os.makedirs(path)

		# print "Started PCA", num_components
		x_train_red, x_test_red = pca.my_PCA(x_train, x_test, num_components)
		# print "PCA Done"

		# print "Sanity Check: Reduced Data"
		# print "x_train: ", x_train_red.shape
		# print "x_test: ", x_test_red.shape

		with open(path + "x_train_red", "wb") as f:
			pickle.dump(x_train_red, f)

		with open(path + "x_test_red", "wb") as f:
			pickle.dump(x_test_red, f)

def classify_reduced(y_train, y_test, components, dataset, split):

	# acc_file = open("Corel/accuracy.txt", "a")

	for num_components in components:

		# path = "../Features/" + dataset + "/" + split + "/PCA/" + str(num_components) + "/"
		path = "../Features/" + dataset + "/" + "/PCA/" + str(num_components) + "/"
		# print path
		with open(path + "x_train_red", "rb") as f:
			x_train_red = pickle.load(f)

		with open(path + "x_test_red", "rb") as f:
			x_test_red = pickle.load(f)

		x_train_red = preprocessing.scale(x_train_red)
		x_test_red = preprocessing.scale(x_test_red)

		# print x_train_red.shape
		# print x_test_red.shape

		# y_test = np.argmax(y_test, axis = 1)
		# y_train = np.argmax(y_train, axis = 1)

		# model = XGBClassifier()
		# model.fit(x_train_red, y_train)

		# y_pred = model.predict(x_test_red)
		# # y_pred = np.argmax(y_pred, axis = 1)
		# # print y_pred

		# acc = (np.sum(y_pred == y_test))*1.0/(y_test.shape[0])
     	# print acc
		MLP.train_top_model(x_train_red, y_train, x_test_red, y_test, dataset, split, num_components)

		# accuracy = classifiers.KNN(x_train_red, y_train, x_test_red, y_test)
		# # print accuracy
		# acc_file.write(path + ": " + str(accuracy))

	# acc_file.close()
			

dataset = "CIFAR10"
split = "60-40"

x_train, y_train, x_test, y_test = read_data.cifar10()
# print x_train.shape
# print x_test.shape
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# y_train = np.argmax(y_train, axis = 1)
# y_test = np.argmax(y_test, axis = 1)

# y_pred = classifiers.random_forest(x_train, y_train, x_test, y_test)
# # print y_pred
# y_train, y_test = read_data.c1()
# x_train, x_test = read_data.pascal()

MLP.train_top_model(x_train, y_train, x_test, y_test, dataset)

# # print classifiers.SVM(x_train, y_train, x_test, y_test)

# MLP.train_top_model(x_train, y_train, x_test, y_test, dataset, split)

# # print "Sanity Check: Raw Data"
# # print "x_train: ", x_train.shape
# # print "x_test: ", x_test.shape

# x_train = x_train.reshape(x_train.shape[0], -1)
# x_test = x_test.reshape(x_test.shape[0], -1)

# # print "Sanity Check: Reshaped Data"
# # print "x_train: ", x_train.shape
# # print "x_test: ", x_test.shape

# components = [50, 75, 100, 125, 150, 200, 300, 400]
# components = [400]

	
# store_reduced_features(x_train, x_test, components, dataset, split)
# classify_reduced(y_train, y_test, components, dataset, split)

