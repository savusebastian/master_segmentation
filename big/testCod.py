import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
	# config.gpu_options.allow_growth = True
	# session = tf.compat.v1.Session(config=config)
	print('predict test data')
	model = get_unet((256, 256, 3))
	model.load_weights('models/unet.hdf5')
	img_test = Image.open('aerial_image_dataset/validation/images/austin162.png')
	img_test = np.array(img_test.resize((256, 256)))
	plt.figure(1)
	plt.imshow(img_test.astype(dtype=np.uint8), vmin=0, vmax=255)

	img_test_pred = img_test.astype(dtype=np.float32)
	img_test_pred /= 255.0

	img = np.zeros((1, 256, 256, 3))
	img[0, :, :, :] = np.copy(img_test_pred)
	imgs_mask_test = model.predict(img)
	imgs_mask_test = imgs_mask_test[0, :, :, 0]

	mask = np.floor(imgs_mask_test * 255)

	imgs_mask_test[imgs_mask_test <= 0.5] = 0
	imgs_mask_test[imgs_mask_test > 0.5] = 1

	plt.figure(2)
	plt.imshow(imgs_mask_test.astype(dtype=np.uint8), vmin=0, vmax=1)

	for i in range(256):
		for j in range(256):
			if imgs_mask_test[i, j] == 1:
				img_test[i, j, :] = 0

	plt.figure(2)
	plt.imshow(img_test)
	plt.show()




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
	img_test = Image.open('aerial_image_dataset/validation/images/austin162.png').convert('L')
	preds_train = model.predict(img_test, verbose=2)
	# preds_val = model.predict(dataset['val'], verbose=2)
	ix = np.random.randint(len(preds_train))
	# preds_test = model.predict(X_test, verbose=2)

	# Threshold predictions
	threshold = .408
	# preds_train_t = (img > threshold).astype(np.uint8)
	# preds_val_t = (dataset['val'] > threshold).astype(np.uint8)
	# preds_test_t = (preds_test > threshold).astype(np.uint8)

	# ix = np.random.randint(len(preds_val))
	# plot_sample(X_test, y_test, preds_test, preds_test_t, ix=ix)

	threshold = .4082
	binarize = .1
	intersection = np.logical_and(img_test.squeeze() > binarize, preds_train.squeeze() > threshold)
	print('Intersection:', intersection)
	union = np.logical_or(img_test.squeeze() > binarize, preds_train.squeeze() > threshold)
	print('Union:', union)
	iou = np.sum(intersection) / np.sum(union)
	print('IOU:', iou)

	# calc_iou_plot(dataset['val'], dataset['val'])
	# calc_iou_plot(y_valid, y_valid)
	# calc_iou_plot(y_test, y_test)
