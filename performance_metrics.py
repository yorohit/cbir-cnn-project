import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score

def performance(y_test, y_pred):

	accuracy = accuracy_score(y_test, y_pred)
	
	# cm = confusion_matrix(y_test, y_pred)

	# precision = 
	# recall = 
	# f_score = 
	# roc_auc = 
	# pr_auc = 

	return accuracy

def plot():
	
	y_true, y_pred = read_data()

	# y_true_b = label_binarize(y_true, classes = [0, 1])
	# y_pred_b = label_binarize(y_pred, classes = [0, 1])

	precision = dict()
	recall = dict()
	pr_auc = dict()

	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	# precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
	# pr_auc["micro"] = average_precision_score(y_true, y_pred, average="micro")

	# plt.plot(recall["micro"], precision["micro"], color='red', lw=2, label='CSLDP, auc = {0}'.format(pr_auc["micro"]))

	# plt.xlabel('Recall')
	# plt.ylabel('Precision')	
	# plt.ylim([0.0, 1.05])
	# plt.xlim([0.0, 1.0])
	# plt.legend(loc="upper right")
	# plt.title('Precision Recall Curve ')
	# plt.show()

	fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
	roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

	print(roc_auc["micro"])

	plt.plot(fpr["micro"], tpr["micro"], label= 'auc = {0}'.format(roc_auc["micro"]))

	plt.plot([0, 1], [0, 1], color='gold', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic ')
	# plt.legend(loc="lower right")
	plt.show()
	