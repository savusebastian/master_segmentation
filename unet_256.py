import os

from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np
import tensorflow as tf

from unet_model import *


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
	plt.title(f'Training Thresh vs IoU {threshold_best}, {iou_best}')
	plt.plot(thresholds, ious)
	plt.plot(threshold_best, iou_best, label='Best threshold')
	plt.xlabel('Threshold')
	plt.ylabel('IoU')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	with tf.device('/device:gpu:0'):
		# Get all image files
		images = os.listdir('aerial_images_256/acoperis')
		np.random.shuffle(images)
		X = np.zeros((len(images), 128, 128, 1), dtype=np.float32)
		y = np.zeros((len(images), 128, 128, 1), dtype=np.float32)
		index = 0
		print('Number of images:', len(images))

		# Convert images & masks to arrays
		for image in images:
			img_orig = np.array(Image.open(f'aerial_images_256/acoperis/{image}').convert('L').resize((128, 128)))
			x_img = np.reshape(img_orig, (128, 128, 1))
			mask_orig = np.array(Image.open(f'aerial_images_256/acoperis_masks/{image}').convert('L').resize((128, 128)))
			mask = np.reshape(mask_orig, (128, 128, 1))

			X[index] = x_img / 255.0
			y[index] = mask / 255.0
			index += 1

		# Split data
		train_index = int(0.8 * len(X)) - 1
		valid_index = len(X) - int(0.9 * len(X))
		X_train, X_valid, X_test = X[:train_index], X[train_index:train_index + valid_index], X[train_index + valid_index:]
		y_train, y_valid, y_test = y[:train_index], y[train_index:train_index + valid_index], y[train_index + valid_index:]

		input_img = Input((128, 128, 1), name='img')
		model = get_unet(input_img, n_filters=16, dropout=0.05)
		model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
		model.summary()

		callbacks = [
			EarlyStopping(patience=15, verbose=2),
			ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=2),
			ModelCheckpoint('models/models_256/model.h5', verbose=2, save_best_only=True, save_weights_only=True)
		]

		results = model.fit(X_train, y_train, batch_size=16, epochs=60, callbacks=callbacks, validation_data=(X_valid, y_valid))

		# Plot Loss vs Epoch
		# plt.figure()
		# plt.title('Learning curve')
		# plt.plot(results.history['loss'], label='loss')
		# plt.plot(results.history['val_loss'], label='val_loss')
		# plt.plot(np.argmin(results.history['val_loss']), np.min(results.history['val_loss']), marker='x', label='best model')
		# plt.xlabel('Epochs')
		# plt.ylabel('log_loss')
		# plt.legend()
		# plt.show()

		# Plot Accuracy vs Epoch
		# acc = results.history['acc']
		# val_acc = results.history['val_acc']
		# epochs = range(len(acc))
		# plt.figure()
		# plt.title('Training and validation accuracy')
		# plt.plot(epochs, acc, label='Training acc')
		# plt.plot(epochs, val_acc, label='Validation acc')
		# plt.legend()
		# plt.show()

		# Load the best model
		model.load_weights('models/models_256/model_8_3.h5')

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

		# threshold = .4082
		# binarize = .1
		# intersection = np.logical_and(y_test[ix].squeeze() > binarize, preds_test[ix].squeeze() > threshold)
		# union = np.logical_or(y_test[ix].squeeze() > binarize, preds_test[ix].squeeze() > threshold)
		# iou = np.sum(intersection) / np.sum(union)
		# print('IOU:', iou)
		#
		# calc_iou_plot(y_train, y_valid)
		# calc_iou_plot(y_valid, y_valid)
		# calc_iou_plot(y_test, y_test)
