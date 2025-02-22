# -*- coding: utf-8 -*-
"""Pure_VGG16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1330rGRxiaFnmmjm6QhxB26W2x9x_5vEr
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

# Load the VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = Flatten()(base_model.output)
x = Dense(30, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = base_model.input, outputs = x)

# Compile the model with the learning_rate parameter
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Paths to your training and testing datasets
train_data_dir = '/content/drive/MyDrive/Asamyuktha_Hastas/Training Images'
test_data_dir = '/content/drive/MyDrive/Asamyuktha_Hastas/Testing Images'

# Create an ImageDataGenerator for data augmentation and loading data from directories
datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]
)

# Load and augment training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

# Load and augment testing data
test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Fit the model using the training data
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=20)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test loss: {loss*100:.2f}%')
print(f'Test accuracy: {accuracy*100:.2f}%')

model.save("/content/drive/MyDrive/Asamyuktha_Hastas/base_vgg16.h5")