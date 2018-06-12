from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten





def my_CNN(input_shape, num_classes,f_sz,k_sz):
    model = Sequential()
    model.add(Convolution2D(filters=f_sz, kernel_size=(k_sz, k_sz), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(k_sz, k_sz),strides=(128,128), padding='same'))

    model.add(Activation('relu'))
    model.add(Flatten())\
    model.add(Activation('softmax',name='predictions'))
    return model

def my_CNN3(input_shape, num_classes,f_sz,k_sz):
#   insert your model here
    return model



def my_CNN7(input_shape, num_classes,f_sz,k_sz):
#   insert your model here
    return model

def mVGG19(input_shape, num_classes):
    base_model = VGG19(weights = None, include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model



if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = my_CNN(input_shape, num_classes,128,128)   
    model.summary()
