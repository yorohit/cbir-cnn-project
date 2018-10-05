import numpy as np

import read_data

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications, regularizers

from sklearn.metrics import accuracy_score

import cPickle

def create_model(train_shape):

	model = Sequential()
	model.add(Flatten(input_shape = train_shape[1:]))
	# model.add(Dense(200, input_shape = train_shape[1:], activation = 'relu', activity_regularizer = regularizers.l2(0.1)))
	model.add(Dense(400, activation = 'relu'))
	# model.add(Dropout(0.7))
    # model.add(Dense(4096, activation = 'relu'))
    # model.add(Dropout(0.25))
	model.add(Dense(20, activation = 'softmax'))

    # model = Sequential()
    # model.add(Flatten(input_shape = train_shape[1:]))
    # model.add(Dense(400, activation = 'relu'))
    # model.add(Dense(20, activation = 'softmax'))

	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def evaluate(x_test, y_test):

	model = create_model(x_test.shape)
	model.load_weights("ghim")

	y_pred = (model.predict(x_test))
	y_pred = np.argmax(y_pred, axis = 1)
	y_test = np.argmax(y_test, axis = 1)

	acc = (np.sum(y_pred == y_test))*1.0/(y_test.shape[0])

	print y_test[50:100]
	# acc = accuracy_score(y_test, y_pred)
	print acc


dataset = "GHIM-10K"
split = "90-10"
num_components = 100

# path = "../Features/" + dataset + "/" + split + "/PCA/" + str(num_components) + "/"

# with open(path + "x_train_red", "rb") as f:
# 	x_train = cPickle.load(f)

# with open(path + "x_test_red", "rb") as f:
# 	x_test = cPickle.load(f)

_, _, x_test, y_test = read_data.ghim()

evaluate(x_test, y_test)