#!/home/abhinav/tensorflow/bin/python
import imp
imp.load_source('cnn',"../../models/cnn.py")
from cnn import my_CNN3,my_CNN7

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import glob
import PIL
import os.path

from keras.models import Sequential
from extract_faces import *



input_shape = (64, 64, 1)
num_classes = 7
emotions_c = ['anger','disgust','fear','happy','sadness','surprise','neutral']




def create_model():
   model = my_CNN3(input_shape,num_classes,32,8)
   return model


def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   return model


src_path="./tImages/"
X_test=[]
src_name=[]
def myDataLoader(fname):
   img = src_path+fname
   if(os.path.isfile(img)):
      print("File Exists!!")
      ldimg = load_img(img,grayscale=True,target_size=(64,64))  # this is a PIL image
      #print ldimg
      x = img_to_array(ldimg)
      print x.shape
      X_test=[]
      X_test.append(x)
      src_name.append(img)
      X_Test= np.array(X_test)
      X_Test /=255
      X_Test = X_Test - 0.5
      X_Test = X_Test * 2.0
      print('X_train shape:',X_Test.shape)
      return X_Test
   else:
      return []
      

modelpath="../../results/cnn3/fer2013_model.22-0.65.hdf5"
model=load_trained_model(modelpath)



basepath="./myImages/";
#basepath="~/Desktop/imgfolder"
while(1):
   print " enterpath to input file"
   src_name=[]
   strinput = raw_input()
   if(os.path.isfile(basepath+strinput)):
      print("File Exists!!")
   else:
      continue
   detect_faces(basepath+strinput)
   faces = myDataLoader(strinput)
   if len(faces)==0:
	print strinput + " has no face"
	continue
   Y_pred = model.predict(faces)
   y_pred = np.argmax(Y_pred,axis=1)
   print y_pred
   for i in range(0,len(y_pred)):
      print src_name[i]
      print emotions_c[y_pred[i]]
print "end"
