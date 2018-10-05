import numpy as np

import read_data, classifiers, pca, MLP

from sklearn.model_selection import train_test_split

import cPickle, os


x_train = np.load(open('/home/rohit/Image Datasets/VOC2012/temp/60-40split/pascal_cnn_train.npy'))
x_test = np.load(open('/home/rohit/Image Datasets/VOC2012/temp/60-40split/pascal_cnn_test.npy'))

y_train = np.load(open('/home/rohit/Image Datasets/VOC2012/temp/60-40split/pascal_labels_train.npy'))
y_test = np.load(open('/home/rohit/Image Datasets/VOC2012/temp/60-40split/pascal_labels_test.npy'))

print y_train.shape

# y_train, y_test = read_data.c1()

MLP.train_top_model(x_train, y_train, x_test, y_test)