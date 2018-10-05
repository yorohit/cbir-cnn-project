import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import read_data

# dimensions of our images.
img_width, img_height = 384, 256

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/home/rohit/pro/Data/C1K/90-10/train'
validation_data_dir = '/home/rohit/pro/Data/C1K/90-10/val'
nb_train_samples = 900
nb_validation_samples = 100

epochs = 50
batch_size = 10
numClass = 20

def save_bottlebeck_features():
    
    print "build the VGG16 network"    
    datagen = ImageDataGenerator(rescale=1. / 255)

    print "build the VGG16 network"
    model = applications.VGG16(include_top=False, weights='imagenet')
    print "model"
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print "generator train done"
    
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples / batch_size)
    
    np.save(open('/home/rohit/pro/Features/C1K/90-10/c1_cnn_train.npy', 'w'),
            bottleneck_features_train)
    
    print "predict_generator done"
    
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    
    np.save(open('/home/rohit/pro/Features/C1K/90-10/c1_cnn_test.npy', 'w'),
            bottleneck_features_validation)

def fully_connected_model(train_shape):

    model = Sequential()
    model.add(Flatten(input_shape = train_shape[1:]))
    model.add(Dense(200, activation = 'relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(200, activation = 'relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(20, activation = 'softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_top_model():
    
    print "Entered function"
    
    x_train, y_train, x_test, y_test = read_data.read_data()

    model = fully_connected_model(x_train.shape)

    model.summary()
    
    print "Started training"
    
    model.fit(x_train, x_test,
              epochs = epochs,
              batch_size = batch_size,
              validation_data = (y_train, y_test))

    print "Training done"
    
    model.save_weights(top_model_weights_path)
    
    print "Saved weights"

save_bottlebeck_features()
# train_top_model()