import numpy as np

import keras.backend as K

from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, regularizers
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

import read_data

def fully_connected_model(train_shape):

    model = Sequential()
    model.add(Flatten(input_shape = train_shape[1:]))
    # model.add(Dense(80, activation = 'relu'))
    # model.add(Dense(50, activation = 'relu'))
    # model.add(Dense(10, activation = 'softmax'))#, activity_regularizer = regularizers.l2(0.001)))
    
    model.add(Dense(10, activation = 'softmax'))# , activity_regularizer = regularizers.l2(0)))
    # model.add(Dense(10, input_shape = train_shape[1:], activation = 'softmax'))#, activity_regularizer = regularizers.l2(0.005)))
    # model.add(Dense(10, activation = 'softmax'))

    # model.add(Dense(100, activation = 'relu'))
    # model.add(Dropout(0.7))
    # model.add(Dense(4096, activation = 'relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(20, activation = 'softmax'))

    # model = Sequential()
    # model.add(Flatten(input_shape = train_shape[1:]))
    # model.add(Dense(400, activation = 'relu'))
    # model.add(Dense(100, activation = 'softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_top_model(x_train, y_train, x_test, y_test, dataset, split = None, num_components = None):
    
    # print "Entered function"
    # print x_test.shape
    
    # top_model_weights_path = "Models/" + dataset + "/" + split + "/original"
    # top_model_weights_path = "Models/" + dataset + "/" + split + "/" + str(num_components)
    top_model_weights_path = "Models/" + dataset + "/original"
    # top_model_weights_path = "Models/" + dataset + "/" + str(num_components)

    epochs = 1
    batch_size = 16
    numClass = 10

    # x_train, y_train, x_test, y_test = read_data.read_data()

    model = fully_connected_model(x_train.shape)
    # model = load_model(top_model_weights_path)

    stop_here_please = EarlyStopping(patience = 7)

    model.summary()
    
    # print "Started training"
    
    model.fit(x_train, y_train,
              epochs = epochs,
              batch_size = batch_size,
              validation_data = (x_test, y_test))

    # model.fit(x_train, y_train,
    #           epochs = epochs,
    #           batch_size = batch_size,
    #           validation_data = (x_test, y_test), 
    #           callbacks = [stop_here_please])

    y_pred = (model.predict(x_test))
    # y_pred = (y_pred > 0.5).astype(int)
    # y_pred = np.argmax(y_pred, axis = 1)
    # y_test = np.argmax(y_test, axis = 1)

    positive = 0
    for i in (y_pred == y_test):
        temp = 1
        for j in i:
            temp = temp and j
        positive += temp        

    # acc = positive * 1.0 / (y_test.shape[0])
    acc = (np.sum(y_pred == y_test))*1.0/(y_test.shape[0])
    # print acc

    # print "Training done"
    
    model.save(top_model_weights_path)
    