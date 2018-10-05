import os
import numpy as np
import cPickle
from collections import defaultdict
import cPickle, cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from skimage import io
from PIL import Image

path = "/home/rohit/pro/pascal_images_split/60-40/val_split"

def load_data(folder):
	
	x = np.load(open(folder))
	return x

def create_folder(path):
	
	for i in range(0,20):

		if not os.path.exists(path +"/"+ str(i)):
			os.makedirs(path + "/"+str(i))

def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    # cv2.imshow('image',images[29])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return images

def load_labels(folder):
	with open(folder, 'rb') as f:
		label = cPickle.load(f)
	return label

def load_labels1(folder):
	y = np.load(open(folder))
	return y

def process(path):
	
	# create_folder(path, split, components)
	create_folder(path)

	# with open("/home/rohit/pro/Features/Pascal/" + split + "/PCA/" + str(components) + "/x_train_red", "rb") as input_file:
	# 	x = cPickle.load(input_file)

	#input_file = "/home/rohit/pro/pascal_images_split/60-40/train"
	#x = load_images(input_file)
	X = load_images("/home/rohit/pro/Rotated/VOC2012")
	x=X[10274:]


	#y = load_data("/home/rohit/pro/Features/Pascal/" + split + "/pascal_labels_train.npy")
	#y= load_labels(os.path.join('/home/rohit/Image Datasets/VOC2012/annotation',"one_hot_label"))
	y= load_labels1('/home/rohit/pro/pascal_images_split/60-40/bottleneck_labels_test.npy')
	#storage_dict = defaultdict(list)
	
	for index,y_i in enumerate(y):
		for y_i_j,vl in enumerate(y_i):
			#print index,len(y_i)
			img=x[index]
			if(vl == 1):
				# storage_dict[path + "Pascal/" + split + "/" + str(components) + "/" + str(y_i_j)].append(x[index])
				#storage_dict[path + "Pascal/" + split + "/original/" + str(y_i_j)].append(x[index])
				#print index,y_i_j
				# cv2.imshow('image',x[29])
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				#print y_i_j
				cv2.imwrite(path+'/'+str(y_i_j)+'/'+str(index)+'.jpg',x[index])
				#print "yo ",index,y_i_j


	#print storage_dict[0]


	# for i in range(0,20):
		
	# 	file_pi = open(path + "Pascal/" + split + "/original/" + str(i) + "/data", 'w') 
	# 	cPickle.dump(storage_dict[path + "Pascal/" + split + "/original/" + str(i)], file_pi)
	# 	file_pi.close()

#split = "80-20"
# num_components = [50, 75, 100, 125, 150, 200, 300, 400]
# for components in num_components:
process(path)	
#create_folder(path)
#x=load_data("/home/rohit/pro/Features/Pascal/90-10/pascal_cnn_train.npy")











