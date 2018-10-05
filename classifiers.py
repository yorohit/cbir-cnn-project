import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

import performance_metrics

import cPickle

def KNN(x_train, y_train, x_test, y_test):

	neigh = KNeighborsClassifier(n_neighbors = 21)
	neigh.fit(x_train, y_train)

	y_pred = neigh.predict(x_test)

	return performance_metrics.performance(y_test, y_pred)

def MLP(x_train, y_train, x_test, y_test):

	clf = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000, alpha=0.000001, solver='sgd', verbose=10, random_state=42, tol=0.000000001)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)

	performance(y_test, y_pred)

def random_forest(x_train, y_train, x_test, y_test):

	rng = np.random.RandomState(42)
	clf = RandomForestClassifier(random_state=rng)
	clf.fit(x_train, y_train)
	
	y_pred = clf.predict(x_test)
	# y_pred = [int(i) for i in y_pred]

	return y_pred
	# performance_metrics.performance(y_test, y_pred)

def xgb(x_train, y_train, x_test, y_test):

	model = XGBClassifier()
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
		

	performance_metrics.performance(y_test, y_pred)

def SVM(x_train, y_train, x_test, y_test):

	clf = SVC(kernel = 'rbf')
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)

	print performance_metrics.performance(y_test, y_pred)

	with open('SVM_cifar', 'wb') as f:
		cPickle.dump(clf, f)

	# with open('SVM_cifar', 'rb') as f:
	# 	clf = cPickle.load(f)


 # def my_MLP(x_train, y_train, x_test, y_test):



