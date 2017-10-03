from keras.layers import Activation, Reshape, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense
from keras.models import Sequential


def getModel() -> Sequential:
    # training is done on images of 32x32 pixels
    input_width=32
    input_height=32
    # CNN for classification
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1',padding='SAME', input_shape=(input_width, input_height, 3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',padding='SAME', name='conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#model.add(Conv2D(32, (3, 3), activation='relu',padding='SAME', name='conv3'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # the size of this convolution has to be the size of the output at this point - image is 32x32
    model.add(Conv2D(32, (8, 8), activation='relu',padding='VALID', name='conv4'))
   
    model.add(Conv2D(2, (1, 1), activation='linear',padding='VALID', name='final'))
  
    model.add(Flatten())

    model.add(Activation('softmax'))
    return model

def getSegModel(input_width, input_height) -> Sequential:
    # CNN for semantic segmentation - we're going to use the trained weights from the classifier so this matches the getModel architecture 
    modelSeg = Sequential()
    modelSeg.add(Conv2D(32, (3, 3), activation='relu', name='conv1',padding='SAME', input_shape=(input_height,input_width,  3)))
    modelSeg.add(MaxPooling2D((2, 2), strides=(2, 2)))

    modelSeg.add(Conv2D(32, (3, 3), activation='relu',padding='SAME', name='conv2'))
    modelSeg.add(MaxPooling2D((2, 2), strides=(2, 2)))

#modelSeg.add(Conv2D(32, (3, 3), activation='relu',padding='SAME', name='conv3'))
#modelSeg.add(MaxPooling2D((2, 2), strides=(2, 2)))

    modelSeg.add(Conv2D(32, (8, 8), activation='relu',padding='SAME', name='conv4'))

    modelSeg.add(Conv2D(2, (1, 1), activation='linear',padding='SAME', name='final'))


    _, curr_width, curr_height, curr_channels = modelSeg.layers[-1].output_shape

    modelSeg.add(Reshape((curr_width * curr_height, curr_channels)))
    modelSeg.add(Activation('softmax'))
    return modelSeg

