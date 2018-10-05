import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers


def normalize(x_train, x_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(x_train, axis=(0,1,2,3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std+1e-7)
        x_test = (x_test - mean) / (std+1e-7)
        return x_train, x_test

def get_output(model, x_train, i):

    get_ith_layer_output = K.function([model.layers[0].input], [model.layers[i].output])
    layer_output = get_ith_layer_output([x_train])[0]
    return layer_output

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


mean = 120.707
std = 64.15

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train, x_test = normalize(x_train, x_test)

model = load_model('model')

bot_train = get_output(model, x_train[0:5000], 51)
# bot_test = get_output(model, x_test, 51)

np.save(open('bot_train_1', 'wb'), bot_train)
# np.save(open('bot_test', 'wb'), bot_test)

