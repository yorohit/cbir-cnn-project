#import cv2
#import glob
#for img in glob.glob("/home/rohit/Image Datasets/VOC2012/All Old Images/*.jpg"):
#    cv_img = cv2.imread(img)

import matplotlib.image as mpimg
from PIL import Image
import os

width = 256
height= 256

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def load_images1(folder):
    images = []
    for filename in os.listdir(folder):
        #img = mpimg.imread(os.path.join(folder, filename))
        im=Image.open(os.path.join(folder, filename))
        im = im.resize((width,height))
        #im.show()
        im.save(os.path.join('/home/rohit/Image Datasets/VOC2012/All Old Images1', filename))


img=load_images1('/home/rohit/Image Datasets/VOC2012/All Old Images')


'''result=[]
for imgfile in img:
	im=Image.open(imgfile)
	im = im.resize((basewidth,hsize))
	print im.shape()
	result.append(im)
'''


