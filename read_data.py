import numpy as np

from sklearn import preprocessing

from keras.datasets import cifar10 as c

from sklearn.preprocessing import label_binarize

import itertools

def c10(split):

	num_class = 100
	num_train_per_class = int(int(split[0:2]) * 0.01 * 100)
	num_test_per_class = int(int(split[3:5]) * 0.01 * 100)

	train_file = "/home/rohit/pro/Features/C10K/" + split + "/c10_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/C10K/" + split + "/c10_cnn_test.npy"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test
	# return y_train, y_test

def c1(split):

	num_class = 10
	
	num_train_per_class = int(int(split[0:2]) * 0.01 * 100)
	num_test_per_class = int(int(split[3:5]) * 0.01 * 100)

	train_file = "/home/rohit/pro/Features/C1K/" + split + "/c1_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/C1K/" + split + "/c1_cnn_test.npy"

	print "OK"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test
	# return y_train, y_test
	# print x_train.shape
	# return y_train, y_test

def pascal(split):

	num_class = 20

	train_file = "/home/rohit/pro/Features/Pascal/" + split + "/pascal_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/Pascal/" + split + "/pascal_cnn_test.npy"

	# x_train = np.load(open(train_file))
	# x_test = np.load(open(test_file))

	y_train = np.load(open("/home/rohit/pro/Features/Pascal/" + split + "/pascal_labels_train.npy"))
	y_test = np.load(open("/home/rohit/pro/Features/Pascal/" + split + "/pascal_labels_test.npy"))

	# return x_train, y_train, x_test, y_test
	return y_train, y_test

def ghim(split):

	num_class = 20
	
	num_train_per_class = int(int(split[0:2]) * 0.01 * 500)
	num_test_per_class = int(int(split[3:5]) * 0.01 * 500)

	train_file = "/home/rohit/pro/Features/GHIM-10K/" + split + "/ghim_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/GHIM-10K/" + split + "/ghim_cnn_test.npy"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test
	# return y_train, y_test	

def cifar10():

	num_class = 10
	# num_train_per_class = 5000
	# num_test_per_class = 200

	# train_file = "/home/rohit/pro/Features/CIFAR10/cifar_cnn_train.npy"
	# test_file = "/home/rohit/pro/Features/CIFAR10/cifar_cnn_test.npy"

	# x_train = np.load(open(train_file))
	# x_test = np.load(open(test_file))

	# y_train = np.load(open("/home/rohit/pro/Features/CIFAR10/cifar_cnn_train_label.npy"))
	# y_test = np.load(open("/home/rohit/pro/Features/CIFAR10/cifar_cnn_test_label.npy"))

	(_, y_train), (_, y_test) = c.load_data()
	y_train = label_binarize(y_train, classes = list(np.arange(10)))
	y_test = label_binarize(y_test, classes = list(np.arange(10)))

	path = '../Features/CIFAR10/'

	for i in range(1, 11):
		if i == 1:
			x_train = np.load(path + 'train/bot_train_' + str(i))
		else:
			temp = np.load(path + 'train/bot_train_' + str(i))
			x_train = np.append(x_train, temp, axis = 0)

	for i in range(1, 3):
		if i == 1:
			x_test = np.load(path + 'val/bot_test_' + str(i))
		else:
			temp = np.load(path + 'val/bot_test_' + str(i))
			x_test = np.append(x_test, temp, axis = 0)			

	return x_train, y_train, x_test, y_test

def preprocess(x_train, x_test):

	x_train = preprocessing.scale(x_train)
	x_test = preprocessing.scale(x_test)

	return x_train, x_test