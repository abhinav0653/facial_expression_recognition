"""
File: cnf_matrix.py
Author: Abhinav Agrawal
Email: 2016pcp5430@mnit.ac.in
"""
import imp
imp.load_source('cnn',"../../models/cnn.py")
from cnn import my_CNN3,my_CNN7
from utils.datasets import split_data
from keras.utils import np_utils
from keras.models import Sequential

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import glob
import PIL
import pickle
import sys
import itertools



# parameters
batch_size = 32
num_epochs = 60
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50

#base_path="/home/abhinav/Desktop/fer/code/secondbaseline/trained_models/"

base_path="../../trained_models/"
log_path="./logs/"

class_names=['anger','disgust','fear','happy','sadness','surprise','neutral']

datasets = ['fer2013']
for dataset_name in datasets:
    print('Test dataset:', dataset_name)

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    print num_samples
    print num_classes
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    val_faces, val_emotions = val_data
 



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





if len(sys.argv)==1:
	print "Insufficient arguments"
	exit

if sys.argv[1]=="single":
	filename='onelayer'
	filter_sz=[128,256,512]
	kernel_sz=[2,4,8,16,32,64]
elif sys.argv[1]=="multi":
	filename='multilayer'
	filter_sz=[32]
	kernel_sz=[8]
else:
	filename='vgg'
	filter_sz=[32]
	kernel_sz=[8]

for f_sz in filter_sz:
    for k_sz in kernel_sz:
	if filename=="onelayer":
            model = my_CNN(input_shape, num_classes,f_sz,k_sz)
	elif filename=="multilayer": 
	    model = my_CNN3(input_shape, num_classes,f_sz,k_sz)
	else:
	    model = mVGG19(input_shape,num_classes)
        print model
	weights_path="../../results/cnn3/modelname"
                
	model.load_weights(weights_path)
        model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
	Y_pred=model.predict(val_faces)
	y_pred = np.argmax(Y_pred,axis=1)
	val_emotions = np.argmax(val_emotions,axis=1)
	print y_pred
	print val_emotions

	print y_pred[0]
	print val_emotions[0]
	
	cnf_matrix = confusion_matrix(val_emotions, y_pred)
#	cnf_matrix = cnf_matrix / cnf_matrix.astype(np.float).sum(axis=1)
	plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
#	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')



