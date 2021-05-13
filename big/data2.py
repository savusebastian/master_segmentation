from glob import glob

from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from model import *


SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Image size that we are going to use
IMG_SIZE = 256
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 1


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
	input_image = tf.image.resize(datapoint['image'], (256, 256))
	input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))

	if tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		input_mask = tf.image.flip_left_right(input_mask)

	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask


@tf.function
def load_image_test(datapoint: dict) -> tuple:
	input_image = tf.image.resize(datapoint['image'], (256, 256))
	input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))
	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask


if __name__ == '__main__':
	# print(AUTOTUNE)
	# dataset_path = 'aerial_image_dataset/'
	# training_data = 'training/'
	# val_data = 'validation/'
	#
	# TRAINSET_SIZE = len(glob(dataset_path + training_data + '/images/*.png'))
	# print(f'The Training Dataset contains {TRAINSET_SIZE} images.')
	#
	# VALSET_SIZE = len(glob(dataset_path + val_data + '/images/*.png'))
	# print(f'The Validation Dataset contains {VALSET_SIZE} images.')
	#
	# train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + 'images/*.png', seed=SEED)
	# train_dataset = train_dataset.map(parse_image)
	#
	# val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + 'images/*.png', seed=SEED)
	# val_dataset = val_dataset.map(parse_image)
	#
	# BATCH_SIZE = 8
	#
	# # for reference about the BUFFER_SIZE in shuffle:
	# # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
	# BUFFER_SIZE = 1000
	#
	# dataset = {'train': train_dataset, 'val': val_dataset}
	#
	# # -- Train Dataset --#
	# dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
	# dataset['train'] = dataset['train'].repeat()
	# dataset['train'] = dataset['train'].batch(BATCH_SIZE)
	# dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
	#
	# # -- Validation Dataset --#
	# dataset['val'] = dataset['val'].map(load_image_test)
	# dataset['val'] = dataset['val'].repeat()
	# dataset['val'] = dataset['val'].batch(BATCH_SIZE)
	# dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
	#
	# # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
	# # config.gpu_options.allow_growth = True
	# # session = tf.compat.v1.Session(config=config)
	# EPOCHS = 20
	#
	# STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
	# VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE
	# input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
	#
	# # model = build_unet((256, 256, 3), 1)
	# model = get_unet(input_size)
	# model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
	# # model.summary()
	# # model_checkpoint = ModelCheckpoint('unet-{epoch:02d}.hdf5', monitor='loss', verbose=1)
	# # results = model.fit(
	# # 	dataset['train'],
	# # 	epochs=EPOCHS,
	# # 	steps_per_epoch=STEPS_PER_EPOCH,
	# # 	validation_steps=VALIDATION_STEPS,
	# # 	validation_data=dataset['val'],
	# # 	callbacks=[model_checkpoint])




	# def plot_sample(X, y, preds, binary_preds, ix=None):
	# 	# Function to plot the results
	# 	has_mask = y[ix].max() > 0
	# 	fig, ax = plt.subplots(1, 4, figsize=(20, 10))
	# 	ax[0].imshow(X[ix, ..., 0], cmap='gray')
	#
	# 	if has_mask:
	# 		ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
	#
	# 	ax[0].set_title('Image')
	# 	ax[1].imshow(y[ix].squeeze(), cmap='autumn')
	# 	ax[1].set_title('Mask')
	# 	ax[2].imshow(preds[ix].squeeze(), cmap='autumn', vmin=0, vmax=1)
	#
	# 	if has_mask:
	# 		ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
	#
	# 	ax[2].set_title('Predicted Mask')
	# 	ax[3].imshow(binary_preds[ix].squeeze(), cmap='autumn', vmin=0, vmax=1)
	#
	# 	if has_mask:
	# 		ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
	#
	# 	ax[3].set_title('Predicted Binary Mask')
	# 	plt.show()
	#
	#
	# def calc_iou_plot(train, valid):
	# 	# Make sure mask is binary for IOU calculation
	# 	size = len(valid)
	# 	# y_train_mod = np.zeros((size, im_width, im_height), dtype=np.int32)
	# 	y_train_mod = train.squeeze() > binarize
	#
	# 	# Set up Predicted Mask for IOU calculation
	# 	preds_train_mod = np.zeros((size, 128, 128), dtype=np.int32)
	# 	preds_train_sq = preds_train.squeeze()
	#
	# 	thresholds = np.linspace(0.0001, 1, 50)
	# 	ious = np.zeros(len(thresholds))
	# 	count = 0
	#
	# 	for threshold in thresholds:
	# 		for i in range(size):
	# 			preds_train_mod[i, :, :] = np.where(preds_train_sq[i, :, :] > threshold, 1, 0)
	#
	# 	iou = np.zeros(size)
	#
	# 	for i in range(size):
	# 		intersection = np.logical_and(y_train_mod[i, :, :], preds_train_mod[i, :, :])
	# 		union = np.logical_or(y_train_mod[i, :, :], preds_train_mod[i, :, :])
	# 		iou[i] = np.sum(intersection) / np.sum(union)
	#
	# 	ious[count] = np.mean(iou)
	# 	count += 1
	#
	# 	threshold_best_index = np.argmax(ious)
	# 	iou_best = ious[threshold_best_index]
	# 	threshold_best = thresholds[threshold_best_index]
	#
	# 	plt.figure()
	# 	plt.title(f'Training Thresh vs IoU')
	# 	plt.plot(thresholds, ious)
	# 	plt.plot(threshold_best, iou_best, label='Best threshold')
	# 	plt.xlabel('Threshold')
	# 	plt.ylabel('IoU')
	# 	plt.legend()
	# 	plt.show()
	#
	# # Load the best model
	# model.load_weights('unet-20.hdf5')
	#
	# # Evaluate on validation set (this must be equals to the best log_loss)
	# # model.evaluate(dataset['train'], dataset['val'], verbose=2)
	#
	# # Evaluate on validation set (this must be equals to the best log_loss)
	# # model.evaluate(X_test, y_test, verbose=2)
	#
	# # Predict on train, val and test
	# preds_train = model.predict(dataset['train'], verbose=2)
	# preds_val = model.predict(dataset['val'], verbose=2)
	# ix = np.random.randint(len(preds_val))
	# # preds_test = model.predict(X_test, verbose=2)
	#
	# # Threshold predictions
	# threshold = .408
	# preds_train_t = (dataset['train'] > threshold).astype(np.uint8)
	# preds_val_t = (dataset['val'] > threshold).astype(np.uint8)
	# # preds_test_t = (preds_test > threshold).astype(np.uint8)
	#
	# # ix = np.random.randint(len(preds_val))
	# # plot_sample(X_test, y_test, preds_test, preds_test_t, ix=ix)
	#
	# threshold = .4082
	# binarize = .1
	# intersection = np.logical_and(dataset['val'][ix].squeeze() > binarize, preds_val[ix].squeeze() > threshold)
	# print('Intersection:', intersection)
	# union = np.logical_or(dataset['val'][ix].squeeze() > binarize, preds_val[ix].squeeze() > threshold)
	# print('Union:', union)
	# iou = np.sum(intersection) / np.sum(union)
	# print('IOU:', iou)
	#
	# calc_iou_plot(dataset['val'], dataset['val'])
	# # calc_iou_plot(y_valid, y_valid)
	# # calc_iou_plot(y_test, y_test)




	with tf.device('/device:gpu:0'):
		def plot_sample(X, y, preds, binary_preds, ix=None):
			# Function to plot the results
			has_mask = y[ix].max() > 0
			fig, ax = plt.subplots(1, 4, figsize=(20, 10))
			ax[0].imshow(X[ix, ..., 0], cmap='gray')

			if has_mask:
				ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])

			ax[0].set_title('Image')
			ax[1].imshow(y[ix].squeeze(), cmap='autumn')
			ax[1].set_title('Mask')
			ax[2].imshow(preds[ix].squeeze(), cmap='autumn', vmin=0, vmax=1)

			if has_mask:
				ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])

			ax[2].set_title('Predicted Mask')
			ax[3].imshow(binary_preds[ix].squeeze(), cmap='autumn', vmin=0, vmax=1)

			if has_mask:
				ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])

			ax[3].set_title('Predicted Binary Mask')
			plt.show()


		def calc_iou_plot(train, valid):
			# Make sure mask is binary for IOU calculation
			size = len(valid)
			# y_train_mod = np.zeros((size, im_width, im_height), dtype=np.int32)
			y_train_mod = train.squeeze() > binarize

			# Set up Predicted Mask for IOU calculation
			preds_train_mod = np.zeros((size, 256, 256), dtype=np.int32)
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
			plt.title(f'Training Thresh vs IoU')
			plt.plot(thresholds, ious)
			plt.plot(threshold_best, iou_best, label='Best threshold')
			plt.xlabel('Threshold')
			plt.ylabel('IoU')
			plt.legend()
			plt.show()


		images = os.listdir('aerial_image_dataset/training/images')
		np.random.shuffle(images)
		X = np.zeros((len(images), 256, 256, 1), dtype=np.float32)
		y = np.zeros((len(images), 256, 256, 1), dtype=np.float32)
		index = 0
		print('Number of images:', len(images))

		# Convert images & masks to arrays
		for image in images:
			img_orig = np.array(Image.open(f'aerial_image_dataset/training/images/{image}').convert('L').resize((256, 256)))
			x_img = np.reshape(img_orig, (256, 256, 1))
			mask_orig = np.array(Image.open(f'aerial_image_dataset/training/gt/{image}').convert('L').resize((256, 256)))
			mask = np.reshape(mask_orig, (256, 256, 1))

			X[index] = x_img / 255.0
			y[index] = mask / 255.0
			index += 1

		# Split data
		train_index = int(0.8 * len(X)) - 1
		valid_index = len(X) - int(0.9 * len(X))
		X_train, X_valid, X_test = X[:train_index], X[train_index:train_index + valid_index], X[train_index + valid_index:]
		y_train, y_valid, y_test = y[:train_index], y[train_index:train_index + valid_index], y[train_index + valid_index:]

		input_img = (256, 256, 1)
		model = get_unet(input_img)
		model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
		model.summary()

		callbacks = [
			EarlyStopping(patience=15, verbose=2),
			ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=2),
			ModelCheckpoint('../models/models_256/model.h5', verbose=2, save_best_only=True, save_weights_only=True)
		]

		results = model.fit(X_train, y_train, batch_size=16, epochs=60, callbacks=callbacks, validation_data=(X_valid, y_valid))

		# # Plot Loss vs Epoch
		# plt.figure()
		# plt.title('Learning curve')
		# plt.plot(results.history['loss'], label='loss')
		# plt.plot(results.history['val_loss'], label='val_loss')
		# plt.plot(np.argmin(results.history['val_loss']), np.min(results.history['val_loss']), marker='x', label='best model')
		# plt.xlabel('Epochs')
		# plt.ylabel('log_loss')
		# plt.legend()
		# plt.show()
		#
		# # Plot Accuracy vs Epoch
		# epochs = range(len(results.history['accuracy']))
		# plt.figure()
		# plt.title('Training and validation accuracy')
		# plt.plot(epochs, results.history['accuracy'], label='Training acc')
		# plt.plot(epochs, results.history['val_accuracy'], label='Validation acc')
		# plt.legend()
		# plt.show()

		# Load the best model
		model.load_weights('../models/models_256/model.h5')

		# Evaluate on validation set (this must be equals to the best log_loss)
		model.evaluate(X_valid, y_valid, verbose=2)

		# Evaluate on validation set (this must be equals to the best log_loss)
		model.evaluate(X_test, y_test, verbose=2)

		# Predict on train, val and test
		preds_train = model.predict(X_train, verbose=2)
		preds_val = model.predict(X_valid, verbose=2)
		preds_test = model.predict(X_test, verbose=2)

		# Threshold predictions
		threshold = .408
		preds_train_t = (preds_train > threshold).astype(np.uint8)
		preds_val_t = (preds_val > threshold).astype(np.uint8)
		preds_test_t = (preds_test > threshold).astype(np.uint8)

		ix = np.random.randint(len(preds_val))
		plot_sample(X_test, y_test, preds_test, preds_test_t, ix=ix)

		threshold = .4082
		binarize = .1
		intersection = np.logical_and(y_test[ix].squeeze() > binarize, preds_test[ix].squeeze() > threshold)
		union = np.logical_or(y_test[ix].squeeze() > binarize, preds_test[ix].squeeze() > threshold)
		iou = np.sum(intersection) / np.sum(union)
		print('IOU:', iou)

		calc_iou_plot(y_train, y_valid)
		calc_iou_plot(y_valid, y_valid)
		calc_iou_plot(y_test, y_test)
