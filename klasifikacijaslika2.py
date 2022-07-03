from PIL import Image
import os
import random
import os.path
import cv2
from PIL import ImageEnhance
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

img_size = 500

categories = ['Dog', 'Cat']

data = []

for category in categories:
    folder = os.path.join('./', category)
    label = categories.index(category)
    for img in os.listdir(folder):
        image_path = os.path.join(folder, img)
        img_arr = cv2.imread(image_path)
        print(img)
        #img_arr = cv2.resize(img_arr, (img_size, img_size))
        data.append([img_arr, label])

random.shuffle(data)

x = []
y = []

for features, labels in data:
   x.append(features)
   y.append(labels)

x = np.asarray(x)
y = np.asarray(y)

x = x/255

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, input_shape=x.shape[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x, y, epochs=10, validation_split=0.08)

print(history.history.keys())
 # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

 # TEST

test_image = cv2.imread('dog_test.jpg')
test_img = plt.imshow(test_image)
test_img = cv2.resize(test_image, (img_size, img_size))
predictions1 = model.predict(np.array([test_img]))
print(predictions1)

test_image = cv2.imread('./cat_test.JPG')
test_img = plt.imshow(test_image)
test_img = cv2.resize(test_image, (img_size, img_size))
predictions2 = model.predict(np.array([test_img]))
print(predictions2)