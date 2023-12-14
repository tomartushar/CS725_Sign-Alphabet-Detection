
import numpy as np
from  matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as mpl
from modeling import get_data

def matrix_heatmap(data):
	axis = sns.heatmap(data, linewidth=0.5)
	l1=['Nothing']
	for i in range(26):
		l1.append(chr(97 + i))
	l1.append('del')
	l1.append('space')
	plt.xticks(range(29),l1)
	plt.yticks(range(29),l1)
	axis.tick_params(axis='x', pad=15)
	axis.tick_params(axis='y', pad=15)
	mpl.rcParams['xtick.major.pad'] = 12
	mpl.rcParams['ytick.major.pad'] = 12
	plt.setp(axis.get_yticklabels(), rotation=30, horizontalalignment='right')

	plt.show()
	plt.save('heatmap.png')
	
def get_heatmap(img_folder = 'data/test'):
	test_X, test_Y = get_data(img_folder)
	model_ = load_model('model_s1_2.h5')
	y_pred = np.argmax(model_.predict(test_X), axis=-1)
	cm = confusion_matrix(test_Y, y_pred)
	matrix_heatmap(cm)
	

if __name__ == '__main__':
	# get_heatmap()
	pass