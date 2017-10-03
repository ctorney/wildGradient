
import numpy as np
from keras import callbacks, optimizers
from keras.callbacks import ModelCheckpoint

from deepModels import getModel
from utils import dataset
batch_size = 16

model = getModel()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


#Prepare input data
classes = ['no','yes']

num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 32
num_channels = 3
train_path='training/'
filepath='training/weights.{epoch:02d}-{loss:.2f}.hdf5'

cb = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data.train.images, data.train.labels, epochs=1000, batch_size=128,callbacks=[cb],verbose=0)
filepath='training/weights.hdf5'
model.save_weights(filepath)


