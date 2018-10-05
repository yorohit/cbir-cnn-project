import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import cPickle 
from sklearn.model_selection import train_test_split
import os
import matplotlib.image as mpimg
from PIL import Image
#import pandas as pd


def load_images1(folder):
	imglis=[]
	for filename in os.listdir(folder):
		im=mpimg.imread(os.path.join(folder, filename))
		if im is not None:
			imglis.append(im)
	return imglis

def load_labels(folder):
	with open(folder, 'rb') as f:
		label = cPickle.load(f)
	return label

def split(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
	return X_train, X_test, y_train, y_test


X=load_images1('/home/rohit/Image Datasets/VOC2012/All Old Images1')
Y=load_labels(os.path.join('/home/rohit/Image Datasets/VOC2012/annotation',"one_hot_label"))
X=np.asarray(X)
Y=np.asarray(Y)

X_train, X_test, y_train, y_test = split(X,Y)



print X_train.shape, y_train.shape
print X_test.shape, y_test.shape


