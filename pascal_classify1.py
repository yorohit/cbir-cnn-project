import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import decomposition, cluster
from sklearn.ensemble import RandomForestClassifier

def my_PCA(x_train, x_test):

	pca = decomposition.PCA(n_components = 300)
	pca.fit(x_train)

	x_train = pca.transform(x_train)
	x_test = pca.transform(x_test)

	return x_train, x_test



def load_data(folder):
	x = np.load(open(folder))
	return x

def KNN(x_train, y_train, x_test, y_test):

	neigh = KNeighborsClassifier(n_neighbors = 21)
	neigh.fit(x_train, y_train)

	y_pred = neigh.predict(x_test)

	print accuracy_score(y_test, y_pred)

def MLP(x_train, y_train, x_test, y_test):

	clf = MLPClassifier(hidden_layer_sizes=(400, 300), max_iter=500, alpha=0.000001, solver='adam', verbose=10, random_state=21, tol=0.000000001)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)

	print accuracy_score(y_test, y_pred)

def random_forest(x_train, y_train, x_test, y_test):

	clf = RandomForestClassifier(random_state=42)
	clf.fit(x_train, y_train)

	y_pred = clf.predict(x_test)

	print accuracy_score(y_test, y_pred)


X_train = load_data('/home/rohit/Image Datasets/VOC2012/temp/90-10split/bottleneck_features_train.npy')
y_train = load_data('/home/rohit/Image Datasets/VOC2012/temp/90-10split/bottleneck_labels_train.npy')
X_test  = load_data('/home/rohit/Image Datasets/VOC2012/temp/90-10split/bottleneck_features_test.npy')
y_test  = load_data('/home/rohit/Image Datasets/VOC2012/temp/90-10split/bottleneck_labels_test.npy')


x_train = np.reshape(X_train,(X_train.shape[0],-1))
x_test = np.reshape(X_test,(X_test.shape[0],-1))

print x_train.shape, X_train.shape
print x_test.shape, X_test.shape

print "Started PCA"

x_train, x_test = my_PCA(x_train, x_test)

print "Sanity Check: Reduced Data"
print "x_train: ", x_train.shape
print "x_test: ", x_test.shape

KNN(x_train,y_train,x_test,y_test)
# MLP(x_train,y_train,x_test,y_test)
# random_forest(x_train,y_train,x_test,y_test)