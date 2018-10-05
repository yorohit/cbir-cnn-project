import numpy as np

from keras import applications

model = applications.VGG16(include_top = False, weights = 'imagenet')

model.summary()