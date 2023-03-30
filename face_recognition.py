# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:24:02 2023

@author: hp
"""

from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model,Sequential
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#resize all the images to size
IMAGE_SIZE=[224,224]

train_path="Datasets/train"
test_path="Datasets/test"

#add preprocessing layer to the front of vgg16
vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)

#dont train existing weights
for layer in vgg.layers:
    layer.trainable=False
    
#useful for getting no of classes
folders=glob("Datasets/train/*")

#layers
x=Flatten()(vgg.output)
#x=Dense(1000,activation='relu')(x)
prediction=Dense(len(folders),activation='softmax')(x)

#creating model object
model=Model(inputs=vgg.input,outputs=prediction)
#view the structure of the model
model.summary()

#tell model what cost and optimization function to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set=test_datagen.flow_from_directory("Datasets/test",
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode="categorical")

#fit the model
r=model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
    )

#loss
plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()
plt.savefig('lossVal_loss')

#accuracies
plt.plot(r.history['accuracy'],label='train accuracy')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf
from keras.models import load_model
model.save('facefeature_model.h5')