import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

model=tf.keras.Sequential()


model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3)
padding='same',activation='relu',input_shape=(80,80,1))
mode.add(tf.keras.layers.MaxPooling2D(pool_size=4))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=4))
model.add(tf.keras.layers.Dropout(0.25))


model.add(tf.keras.layers.Flattenn())
model.add(tf.keras.layers.Dense(265,activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.summary()