import numpy as np

import read_data, classifiers, pca, MLP

from sklearn.model_selection import train_test_split

import cPickle, os


x_train = np.load(open('GHIM/ghim_cnn_train.npy'))
x_test = np.load(open('GHIM/ghim_cnn_test.npy'))

y_train, y_test = read_data.ghim()

MLP.train_top_model(x_train, y_train, x_test, y_test)
