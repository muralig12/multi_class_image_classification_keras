#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:55:16 2019

@author: user
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import cv2

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'


# number of epochs to train top model
epochs = 5
# batch size used by flow_from_directory and predict_generator
batch_size = 16

# load the class_indices saved in the earlier step
class_dictionary = np.load('class_indices.npy').item()

num_classes = len(class_dictionary)

    # add the path to your test image below
image_path = './test_images/test2.jpeg'

orig = cv2.imread(image_path)

print("[INFO] loading and preprocessing image...")
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)

# important! otherwise the predictions will be '0'
image = image / 255

image = np.expand_dims(image, axis=0)

# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

# get the bottleneck prediction from the pre-trained VGG16 model
bottleneck_prediction = model.predict(image)

# build top model
model = Sequential()
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights(top_model_weights_path)

# use the bottleneck prediction on the top model to get the final
# classification
class_predicted = model.predict_classes(bottleneck_prediction)

probabilities = model.predict_proba(bottleneck_prediction)

inID = class_predicted[0]

inv_map = {v: k for k, v in class_dictionary.items()}

label = inv_map[inID]

# get the prediction label
print("Image ID: {}, Label: {}".format(inID, label))

# display the predictions with the image
cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

cv2.imshow("Classification", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
