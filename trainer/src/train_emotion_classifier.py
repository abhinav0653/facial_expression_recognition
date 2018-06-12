"""
File: train_emotion_classifier.py
Author: Abhinav Agrawal
Email: 2016pcp5430@mnit.ac.in
"""
import imp
imp.load_source('cnn',"../../models/cnn.py")
from cnn import my_CNN,my_CNN3,my_CNN7,mVGG19,alexNet

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt
import numpy as np
import glob
import PIL
import pickle
import sys



# parameters
batch_size = 32
num_epochs = 300
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 8
patience = 120

base_path="../../trained_models/"
log_path="./logs/"

datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_model'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [model_checkpoint, csv_logger]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    num_samples, num_classes = emotions.shape
    print num_samples
    print num_classes
    train_data, val_data = split_data(faces, emotions, validation_split)
    print len(train_data[0])
    print len(val_data[0])
    train_faces, train_emotions = train_data
 



# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)



if len(sys.argv)==1:
	print "Insufficient arguments"
	exit

if sys.argv[1]=="single":
	filename='onelayer'
	filter_sz=[32]
	kernel_sz=[32]
elif sys.argv[1]=="multi":
	filename='multilayer'
	filter_sz=[32]
	kernel_sz=[32]
else:
	filename='vgg'
	filter_sz=[32]
	kernel_sz=[32]

for f_sz in filter_sz:
    for k_sz in kernel_sz:
	if filename=="onelayer":
            model = my_CNN(input_shape, num_classes,f_sz,k_sz)
	elif filename=="multilayer": 
            model = my_CNN3(input_shape, num_classes,f_sz,k_sz)
	else:
	    model = mVGG19(input_shape,num_classes)
        print model
        model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

	
	with open(log_path+filename+"/"+filename+"_"+str(f_sz)+"_"+str(k_sz),'w+') as fh:
            model.summary(print_fn=lambda x: fh.write(x+'\n'))
            with open(log_path+filename+"/"+filename+"_"+str(f_sz)+"_"+str(k_sz)+".p",'w+') as fp:
	        tr_history=model.fit_generator(data_generator.flow(train_faces, train_emotions,batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)
                pickle.dump(tr_history.history,fp)

