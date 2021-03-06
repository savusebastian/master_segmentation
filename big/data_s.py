from glob import glob
import os

from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model_s import *


SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Image size that we are going to use
IMG_SIZE = 512
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 1


def calc_iou_plot(train, valid):
	# Make sure mask is binary for IOU calculation
	size = len(valid)
	# y_train_mod = np.zeros((size, im_width, im_height), dtype=np.int32)
	y_train_mod = train.squeeze() > binarize

	# Set up Predicted Mask for IOU calculation
	preds_train_mod = np.zeros((size, 512, 512), dtype=np.int32)
	preds_train_sq = preds_train.squeeze()

	thresholds = np.linspace(0.0001, 1, 50)
	ious = np.zeros(len(thresholds))
	count = 0

	for threshold in thresholds:
		for i in range(size):
			preds_train_mod[i, :, :] = np.where(preds_train_sq[i, :, :] > threshold, 1, 0)

	iou = np.zeros(size)

	for i in range(size):
		intersection = np.logical_and(y_train_mod[i, :, :], preds_train_mod[i, :, :])
		union = np.logical_or(y_train_mod[i, :, :], preds_train_mod[i, :, :])
		iou[i] = np.sum(intersection) / np.sum(union)

	ious[count] = np.mean(iou)
	count += 1

	threshold_best_index = np.argmax(ious)
	iou_best = ious[threshold_best_index]
	threshold_best = thresholds[threshold_best_index]

	plt.figure()
	plt.title(f'Training Thresh vs IoU {threshold_best}, {iou_best}')
	plt.plot(thresholds, ious)
	plt.plot(threshold_best, iou_best, label='Best threshold')
	plt.xlabel('Threshold')
	plt.ylabel('IoU')
	plt.legend()
	plt.show()


def parse_image(img_path: str) -> dict:
	image = tf.io.read_file(img_path)
	image = tf.image.decode_png(image, channels=3)
	image = tf.image.convert_image_dtype(image, tf.uint8)
	mask_path = tf.strings.regex_replace(img_path, 'images', 'gt')
	mask = tf.io.read_file(mask_path)
	# The masks contain a class index for each pixels
	mask = tf.image.decode_png(mask, channels=1)
	#mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
	# Note that we have to convert the new value (0)
	# With the same dtype than the tensor itself

	return {'image': image, 'segmentation_mask': mask}


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
	input_image = tf.cast(input_image, tf.float32) / 255.0
	input_mask = tf.cast(input_mask, tf.float32) / 255.0

	return input_image, input_mask


@tf.function
def load_image_train(datapoint: dict) -> tuple:
	# input_image = tf.image.resize(datapoint['image'], (256, 256))
	# input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))
	input_image = datapoint['image']
	input_mask = datapoint['segmentation_mask']

	if tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		input_mask = tf.image.flip_left_right(input_mask)

	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask


@tf.function
def load_image_test(datapoint: dict) -> tuple:
	# input_image = tf.image.resize(datapoint['image'], (256, 256))
	# input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))
	input_image = datapoint['image']
	input_mask = datapoint['segmentation_mask']
	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask


if __name__ == '__main__':
	print(AUTOTUNE)
	# dataset_path = 'aerial_image_dataset_1024/'
	dataset_path = 'aerial_image_dataset/'
	training_data = 'training/'
	val_data = 'validation/'

	TRAINSET_SIZE = len(glob(dataset_path + training_data + '/images/*.png'))
	print(f'The Training Dataset contains {TRAINSET_SIZE} images.')

	VALSET_SIZE = len(glob(dataset_path + val_data + '/images/*.png'))
	print(f'The Validation Dataset contains {VALSET_SIZE} images.')

	train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + 'images/*.png', seed=SEED)
	train_dataset = train_dataset.map(parse_image)

	val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + 'images/*.png', seed=SEED)
	val_dataset = val_dataset.map(parse_image)

	# Batch size of 2 for 1024
	BATCH_SIZE = 4

	# for reference about the BUFFER_SIZE in shuffle:
	# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
	BUFFER_SIZE = 1000

	dataset = {'train': train_dataset, 'val': val_dataset}

	# -- Train Dataset -- #
	dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
	# dataset['train'] = dataset['train'].repeat()
	dataset['train'] = dataset['train'].batch(BATCH_SIZE)
	dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

	# -- Validation Dataset -- #
	dataset['val'] = dataset['val'].map(load_image_test)
	# dataset['val'] = dataset['val'].repeat()
	dataset['val'] = dataset['val'].batch(BATCH_SIZE)
	dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

	# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
	# config.gpu_options.allow_growth = True
	# session = tf.compat.v1.Session(config=config)
	EPOCHS = 20

	STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
	VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE
	input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

	# model = build_unet(input_size, 1)
	# model = get_unet(input_size)
	model = get_efficientnet_unet(input_size)
	# model = get_efficientnet_as_unet(input_size)

	model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
	model.summary()
	# model_checkpoint = ModelCheckpoint('unet-{epoch:02d}.hdf5', monitor='loss', verbose=1)
	model_checkpoint = ModelCheckpoint('efficientnet.hdf5', monitor='loss', verbose=1)
	# model_checkpoint = ModelCheckpoint('efficientnet_as_unet.hdf5', monitor='loss', verbose=1)
	results = model.fit(dataset['train'], epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=dataset['val'], callbacks=[model_checkpoint])


	# # Model
	# model = model_ion()
	# model.summary()
	#
	# model.compile(
	# 	optimizer='adam',
	# 	loss=tf.keras.losses.BinaryCrossentropy(),
	# 	metrics=['accuracy']
	# )
	#
	# epochs = 50
	# history = model.fit(
	# 	dataset['train'],
	# 	validation_data=dataset['val'],
	# 	epochs=epochs
	# )


	# Plot Loss vs Epoch
	plt.figure()
	plt.title('Learning curve')
	plt.plot(results.history['loss'], label='loss')
	plt.plot(results.history['val_loss'], label='val_loss')
	plt.plot(np.argmin(results.history['val_loss']), np.min(results.history['val_loss']), marker='x', label='best model')
	plt.xlabel('Epochs')
	plt.ylabel('log_loss')
	plt.legend()
	plt.show()

	# Plot Accuracy vs Epoch
	acc = results.history['accuracy']
	val_acc = results.history['val_accuracy']
	epochs = range(len(acc))
	plt.figure()
	plt.title('Training and validation accuracy')
	plt.plot(epochs, acc, label='Training acc')
	plt.plot(epochs, val_acc, label='Validation acc')
	plt.legend()
	plt.show()

	# Calc iou
	model.evaluate(dataset['train'], dataset['val'], verbose=2)
	preds_test = model.predict(dataset['val'], verbose=2)

	ix = np.random.randint(len(dataset['val']))
	threshold = .4082
	binarize = .1
	intersection = np.logical_and(dataset['val'][ix].squeeze() > binarize, preds_test[ix].squeeze() > threshold)
	union = np.logical_or(dataset['val'][ix].squeeze() > binarize, preds_test[ix].squeeze() > threshold)
	iou = np.sum(intersection) / np.sum(union)
	print('IOU:', iou)

	# Comment dataset.repeat() lines
	calc_iou_plot(dataset['train'], dataset['val'])
