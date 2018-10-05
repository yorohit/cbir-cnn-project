import numpy as np

from sklearn import decomposition

def my_PCA(x_train, x_test, num_components = 200):

	pca = decomposition.PCA(n_components = num_components)
	pca.fit(x_train)

	x_train = pca.transform(x_train)
	x_test = pca.transform(x_test)

	return x_train, x_test



