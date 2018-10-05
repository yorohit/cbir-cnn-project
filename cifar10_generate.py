import numpy as np
import cPickle, cv2

from keras import applications
from keras.datasets import cifar10

from sklearn import preprocessing

import matplotlib.pyplot as plt 

from skimage import io

path = '../Data/CIFAR10/'

def rarrange(x):
	res=[]
	for temp in x:
		#temp1=[]
		r=temp[:1024]
		g=temp[1024:2048]
		b=temp[2048:]

		r=r.reshape(32,32)
		g=g.reshape(32,32)
		b=b.reshape(32,32)
		temp1=np.dstack((r,g))
		temp1=np.dstack((temp1,b))
		#temp1 = np.append(temp1,r,axis=2)

		#temp1.append(r)
		#temp1.append(g)
		#temp1.append(b)
		#temp1=np.array(temp1)
		#print temp1.shape

		res.append(temp1)

	res=np.array(res)

	return res


def load_data(path):

	# x_train = np.array([])
	# y_train = np.array([])

	for i in range(1, 6):
		
		temp = cPickle.load(open(path + "data_batch_" + str(i), 'rb'))
		temp_x = temp['data']
		temp_y = temp['labels']

		if i == 1:
			x_train = temp_x
			y_train = temp_y
		else:			
			x_train = np.append(x_train, temp_x, axis = 0)
			y_train = np.append(y_train, temp_y)

	x_train = preprocessing.scale(x_train)

	#x_train = x_train.reshape(50000, 32, 32, 3)
	x_train = rarrange(x_train)

	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(10)))

	temp = cPickle.load(open(path + "test_batch", 'rb'))

	x_test = preprocessing.scale(temp['data'])
	#x_test = x_test.reshape(10000, 32, 32, 3)
	x_test = rarrange(x_test)
	y_test = np.array(temp['labels'])
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(10)))

	return x_train, y_train, x_test, y_test

def load_data1():

	# x_train = np.array([])
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(10)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(10)))
	return x_train, y_train, x_test, y_test


def enlarge():

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	t = []

	for img in x_train:

		temp = cv2.resize(img, (48, 48))			
		t.append(temp)

	x_train = np.stack(t)	

	t = []

	for img in x_test:

		temp = cv2.resize(img, (48, 48))
		t.append(temp)

	x_test = np.stack(t)

	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(10)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(10)))

	return x_train, y_train, x_test, y_test

def create_model():

	model = applications.VGG16(include_top = False, weights = 'imagenet')
	return model

def generate_features(model, x_train, x_test, y_train, y_test):

	train_features = model.predict(x_train)
	test_features = model.predict(x_test)

	# Save features
	np.save(open('../Features/CIFAR10/cifar_cnn_train.npy', 'w'), train_features)
	np.save(open('../Features/CIFAR10/cifar_cnn_test.npy', 'w'), test_features)

	np.save(open('../Features/CIFAR10/cifar_cnn_train_label.npy', 'w'), y_train)
	np.save(open('../Features/CIFAR10/cifar_cnn_test_label.npy', 'w'), y_test)

def brain():
	
	#x_train, y_train, x_test, y_test = load_data(path)
	x_train, y_train, x_test, y_test = enlarge()
	x_train = (x_train * 1.)/ 255
	x_test = (x_test * 1.) / 255
	print x_train.shape
	print x_test.shape
	print y_train.shape
	print y_test.shape

	# test = x_train[45]

	# io.imshow(test)
	# plt.show()

	model = create_model()
	generate_features(model, x_train, x_test, y_train, y_test)

brain()