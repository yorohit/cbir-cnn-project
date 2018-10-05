# 1) Load data
# 2) Enlarge
# 3) Pass through VGG16 & store features
# 4) Train MLP for classification


import numpy as np
import cv2

from keras.datasets import cifar10

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

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(x_train)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

model.fit_generator(
    train_generator,
    samples_per_epoch = 50000,
    nb_epoch = 50,
    validation_data = validation_generator,
    nb_val_samples = 10000)

# img = x_train[0]
# print img.shape
# temp = cv2.resize(img, (42, 42))
# print temp.shape