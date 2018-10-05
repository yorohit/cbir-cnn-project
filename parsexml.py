import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cPickle 

def fun(folder):
	lis=[]
	ref={}
	k=0
	for filename in sorted(os.listdir(folder)):
		DOMTree = xml.dom.minidom.parse(os.path.join(folder, filename))
		collection = DOMTree.documentElement
		objects = collection.getElementsByTagName("object")
		res=set()
		one_hot=[0]*20
		#print str(filename)
		for ob in objects:
			label=ob.getElementsByTagName('name')[0]
			#print "label: %s" % label.childNodes[0].data
			label_component=str(label.childNodes[0].data)
			if ref.get(label_component,'nil')=='nil' :
				ref[label_component]=k
				k=k+1
			res.add(label_component)
			one_hot[ref[label_component]]=1
		res=list(res)
		#print res
		lis.append(one_hot)



		#with open(os.path.join('/home/rohit/Image Datasets/VOC2012/annotation',filename), 'wb') as f:
			#cPickle.dump(res,f)
		#print res
		#print one_hot
	with open(os.path.join('/home/rohit/Image Datasets/VOC2012/annotation',"one_hot_label"), 'wb') as f:
		cPickle.dump(lis,f)


	with open(os.path.join('/home/rohit/Image Datasets/VOC2012/annotation',"ref_dict"), 'wb') as f:
		cPickle.dump(ref,f)

	print lis[1]







fun('/home/rohit/Image Datasets/VOC2012/Annotations')



'''
with open('/home/rohit/Image Datasets/VOC2012/Annotation File/2007_000027.xmlpk', 'rb') as f:
	train_data = cPickle.load(f)

print train_data
'''


