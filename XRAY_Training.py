#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np
import os


# In[3]:


pathNormal = "./chest_xray/train/NORMAL"
pathP = "./chest_xray/train/PNEUMONIA"

x = []
y = []

count = 0
for image_path in os.listdir(pathNormal):
    image = Image.open(os.path.join(pathNormal, image_path))
    image = ImageOps.grayscale(image)
    image = image.resize((200, 200))
    data = np.asarray(image)
    list_data = data.tolist()
    x.append(list_data)
    print(count)
    count += 1
    y.append(0)
    
for image_path in os.listdir(pathP):
    image = Image.open(os.path.join(pathP, image_path))
    image = ImageOps.grayscale(image)
    image = image.resize((200, 200))
    data = np.asarray(image)
    print(data.shape)
    list_data = data.tolist()
    x.append(list_data)
    print(count)
    count += 1
    y.append(1)

print('converting')
x = np.asarray(x)
y = np.asarray(y)


# In[4]:

print('shuffle')
shuffler = np.random.permutation(len(y))
x = x[shuffler]
y = y[shuffler]


# In[5]:

print('valid')
pathNormal = "./chest_xray/val/NORMAL"
pathP = "./chest_xray/val/PNEUMONIA"

x_val = []
y_val = []

count = 0
for image_path in os.listdir(pathNormal):
    image = Image.open(os.path.join(pathNormal, image_path))
    image = ImageOps.grayscale(image)
    image = image.resize((200, 200))
    data = np.asarray(image)
    list_data = data.tolist()
    x_val.append(list_data)
    print(count)
    count += 1
    y_val.append(0)
    
for image_path in os.listdir(pathP):
    image = Image.open(os.path.join(pathP, image_path))
    image = ImageOps.grayscale(image)
    image = image.resize((200, 200))
    data = np.asarray(image)
    print(data.shape)
    list_data = data.tolist()
    x_val.append(list_data)
    print(count)
    count += 1
    y_val.append(1)

x_val = np.asarray(x_val)
y_val = np.asarray(y_val)

x = x / 255.0
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
x_val = x_val / 255.0
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
# In[2]:

print(x[0])
print(x[1])

print('model')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200,200, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
# In[ ]:

print('fit')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x, y, epochs=10, 
                    validation_data=(x_val, y_val))


# In[ ]:




