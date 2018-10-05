'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import itertools
from keras.utils import np_utils

# dimensions of our images.
img_width, img_height = 400, 400

top_model_weights_path = '/bottleneck_fc_model.h5'
train_data_dir = '/train'
validation_data_dir = '/val'
nb_train_samples = 9000
nb_validation_samples = 1000
epochs = 10
batch_size = 10
numClass = 20

def save_bottlebeck_features():
    print "build the VGG16 network"    
    datagen = ImageDataGenerator(rescale=1. / 255)

    print "build the VGG16 network"
    model = applications.VGG16(include_top=False, weights='imagenet')
    # model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
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
    np.save(open('/features/bottleneck_features_train.npy', 'w'),
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
    np.save(open('/features/bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

def train_top_model():
    print "Entered function"
    train_data = np.load(open('/home/rohit/Desktop/classification/features/bottleneck_features_train.npy'))
    # train_labels = np.array(
    #     [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    print "train_data loaded"
    labels = []
    for i in xrange(20):
        labels.append([i]*450)
   
    train_labels = np.array(list(itertools.chain.from_iterable(labels)))
    train_labels = np_utils.to_categorical(train_labels, numClass)

    print "train_labels loaded"
    validation_data = np.load(open('/home/rohit/Desktop/classification/features/bottleneck_features_validation.npy'))
    # validation_labels = np.array(
    #     [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    print "validation_data loaded"
    labels = []
    for i in xrange(20):
        labels.append([i]*50)
    validation_labels = np.array(list(itertools.chain.from_iterable(labels)))
    validation_labels = np_utils.to_categorical(validation_labels, numClass)

    print "validation_labels loaded"
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print "Model compiled"
    model.summary()
    print "Started training"
    model.fit(train_data, train_labels,
               epochs=epochs,
               batch_size=batch_size,
               validation_data=(validation_data, validation_labels))

    print "Training done"
    model.save_weights(top_model_weights_path)
    print "Saved weights"

# save_bottlebeck_features()
train_top_model()
