import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing


# http://download.tensorflow.org/example_images/flower_photos.tgz
data_train = './aerial_image_dataset/training/images'
data_test = './aerial_image_dataset/validation/images'

# data_train = pathlib.Path(data_train)
# data_test = pathlib.Path(data_test)

batch_size = 32
img_height = 256
img_width = 256


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	data_train,
	seed=42,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	data_test,
	seed=42,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

img_augmentation = Sequential(
	[
		preprocessing.RandomRotation(factor=0.15),
		preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
		preprocessing.RandomFlip(),
		preprocessing.RandomContrast(factor=0.1),
	],
	name='img_augmentation',
)

num_classes = 2
inputs = layers.Input(shape=(img_height, img_width, 3))
#inputs = img_augmentation(inputs)
outputs = EfficientNetB0(include_top=False, weights=None, classes=num_classes)(inputs)
outputs = layers.GlobalAveragePooling2D()(outputs)
outputs = layers.Dropout(0.2)(outputs)
outputs = layers.Dense(num_classes, activation='softmax')(outputs)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy']
)

epochs = 50
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs
)
