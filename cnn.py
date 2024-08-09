import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os 

import tensorflow as tf 
from tensorflow import keras 
from keras import utils, layers, Sequential, regularizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = tf.keras.utils.image_dataset_from_directory(
    'C:\\python\\data',
    image_size=(128, 128),  
    batch_size=32
)

data_np = list(data.as_numpy_iterator())
images = np.concatenate([x for x, y in data_np])
labels = np.concatenate([y for x, y in data_np])

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)


model = Sequential()


model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))


model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))


model.add(Flatten())


model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, batch_size=32)  

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


img = cv2.imread("ece.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
img = cv2.resize(img, (128, 128))  
img = img.astype('float32') / 255.0  
img = np.expand_dims(img, axis=0)  #


y_pred = model.predict(img)
y_pred_label = "Kadin" if y_pred[0][0] > 0.5 else "Erkek"

print(f"Resimdeki kişinin cinsiyeti: {y_pred_label}")


plt.imshow(cv2.cvtColor(cv2.imread("ece.png"), cv2.COLOR_BGR2RGB))
plt.title(f"Gerçek Etiket: {y_pred_label}")
plt.axis('off')
plt.show()
