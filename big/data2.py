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

from model import build_unet


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


#https://github.com/ShawDa/unet-rgb/blob/master/unet.py
def get_unet(input_shape):
	inputs = Input(input_shape)

	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
	merge6 = Concatenate(axis=3) ([drop4, up6])
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
	merge7 = Concatenate(axis=3) ([conv3, up7])
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
	merge8 = Concatenate(axis=3) ([conv2, up8])
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
	merge9 = Concatenate(axis=3) ([conv1, up9])
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

	output = Conv2D(1, 1, activation='sigmoid')(conv9)

	return Model(inputs, output)


if __name__ == '__main__':
	print(AUTOTUNE)
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

	BATCH_SIZE = 8

	# for reference about the BUFFER_SIZE in shuffle:
	# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
	BUFFER_SIZE = 1000

	dataset = {'train': train_dataset, 'val': val_dataset}

	# -- Train Dataset --#
	dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
	dataset['train'] = dataset['train'].repeat()
	dataset['train'] = dataset['train'].batch(BATCH_SIZE)
	dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

	# -- Validation Dataset --#
	dataset['val'] = dataset['val'].map(load_image_test)
	dataset['val'] = dataset['val'].repeat()
	dataset['val'] = dataset['val'].batch(BATCH_SIZE)
	dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

	# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
	# config.gpu_options.allow_growth = True
	# session = tf.compat.v1.Session(config=config)
	EPOCHS = 20

	STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
	VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE
	input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

	# model = build_unet((256, 256, 3), 1)
	model = get_unet(input_size)
	model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
	# model.summary()
	# model_checkpoint = ModelCheckpoint('unet-{epoch:02d}.hdf5', monitor='loss', verbose=1)
	# results = model.fit(dataset['train'], epochs=EPOCHS,
	# 	steps_per_epoch=STEPS_PER_EPOCH,
	# 	validation_steps=VALIDATION_STEPS,
	# 	validation_data=dataset['val'],
	# 	callbacks=[model_checkpoint])





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
		preds_train_mod = np.zeros((size, 128, 128), dtype=np.int32)
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

	# Load the best model
	model.load_weights('unet-20.hdf5')

	# Evaluate on validation set (this must be equals to the best log_loss)
	# model.evaluate(dataset['train'], dataset['val'], verbose=2)

	# Evaluate on validation set (this must be equals to the best log_loss)
	# model.evaluate(X_test, y_test, verbose=2)

	# Predict on train, val and test
	preds_train = model.predict(dataset['train'], verbose=2)
	preds_val = model.predict(dataset['val'], verbose=2)
	ix = np.random.randint(len(preds_val))
	# preds_test = model.predict(X_test, verbose=2)

	# Threshold predictions
	threshold = .408
	preds_train_t = (dataset['train'] > threshold).astype(np.uint8)
	preds_val_t = (dataset['val'] > threshold).astype(np.uint8)
	# preds_test_t = (preds_test > threshold).astype(np.uint8)

	# ix = np.random.randint(len(preds_val))
	# plot_sample(X_test, y_test, preds_test, preds_test_t, ix=ix)

	threshold = .4082
	binarize = .1
	intersection = np.logical_and(dataset['val'][ix].squeeze() > binarize, preds_val[ix].squeeze() > threshold)
	print('Intersection:', intersection)
	union = np.logical_or(dataset['val'][ix].squeeze() > binarize, preds_val[ix].squeeze() > threshold)
	print('Union:', union)
	iou = np.sum(intersection) / np.sum(union)
	print('IOU:', iou)

	calc_iou_plot(dataset['val'], dataset['val'])
	# calc_iou_plot(y_valid, y_valid)
	# calc_iou_plot(y_test, y_test)
