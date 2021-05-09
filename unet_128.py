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


if __name__ == '__main__':
	with tf.device('/device:gpu:0'):
		# Get all image files
		# files = os.listdir('aerial_images_128/acoperis')
		# np.random.shuffle(files)
		# X = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
		# y = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
		# index = 0
		# print('Number of images:', len(files))
		#
		# # Convert images & masks to arrays
		# for file in files:
		# 	img_orig = np.array(Image.open(f'aerial_images_128/acoperis/{file}').convert('L'))
		# 	x_img = np.reshape(img_orig, (128, 128, 1))
		# 	mask_orig = np.array(Image.open(f'aerial_images_128/acoperis_masks/{file}').convert('L'))
		# 	mask = np.reshape(mask_orig, (128, 128, 1))
		#
		# X[index] = x_img / 255.0
		# y[index] = mask / 255.0
		# index += 1
		#
		# # Split data
		# train_index = int(0.8 * len(X)) - 1
		# valid_index = len(X) - int(0.9 * len(X))
		# X_train, X_valid, X_test = X[:train_index], X[train_index:train_index + valid_index], X[train_index + valid_index:]
		# y_train, y_valid, y_test = y[:train_index], y[train_index:train_index + valid_index], y[train_index + valid_index:]

		input_img = Input((128, 128, 1), name='img')
		model = get_unet(input_img, n_filters=16, dropout=0.05)
		model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
		model.summary()

		# callbacks = [
		# 	EarlyStopping(patience=15, verbose=2),
		# 	ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=2),
		# 	ModelCheckpoint('models/models_128/model.h5', verbose=2, save_best_only=True, save_weights_only=True)
		# ]
		#
		# results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks, validation_data=(X_valid, y_valid))

		# Load the best model
		model.load_weights('models/models_128/model_8_3.h5')

		# # Evaluate on validation set (this must be equals to the best log_loss)
		# model.evaluate(X_valid, y_valid, verbose=2)
		#
		# # Evaluate on validation set (this must be equals to the best log_loss)
		# model.evaluate(X_test, y_test, verbose=2)
		#
		# # Predict on train, val and test
		# preds_train = model.predict(X_train, verbose=2)
		# preds_val = model.predict(X_valid, verbose=2)
		# preds_test = model.predict(X_test, verbose=2)
		#
		# # Threshold predictions
		# threshold = .408
		# preds_train_t = (preds_train > threshold).astype(np.uint8)
		# preds_val_t = (preds_val > threshold).astype(np.uint8)
		# preds_test_t = (preds_test > threshold).astype(np.uint8)
		#
		# ix = np.random.randint(len(preds_val))
		# plot_sample(X_test, y_test, preds_test, preds_test_t, ix=ix)

		img = np.array(Image.open('img1_1.png'), dtype=np.float32)
		img_gray = np.array(Image.open('img1_1.png').convert('L'), dtype=np.float32)
		img_255 = img_gray / 255.0
		w, h, _ = img.shape
		v = 128
		r = w // v
		c = h // v
		data = np.zeros((r * c, v, v), dtype=np.float32)
		new = np.zeros((r * v, c * v, 1), dtype=np.float32)
		index = 0

		for i in range(r):
			for j in range(c):
				data[index, :, :] = img_255[i * v : (i + 1) * v, j * v : (j + 1) * v]
				index += 1

		index = 0
		threshold = 0.4
		data_bin = (model.predict(data) > threshold).astype(np.uint8)

		for i in range(r):
			for j in range(c):
				# print(data_bin[index])
				new[i * v : (i + 1) * v, j * v : (j + 1) * v, :] = data_bin[index]
				index += 1

		new = np.reshape(new * 255, (r * v, c * v))
		new_image = Image.fromarray(new, 'L')
		new_image.save('img_p.jpg')

	# print('batch_size: 8 - val_acc: 0.946')
	# print('batch_size: 16 - val_acc: 0.946')
	# print('batch_size: 32 - val_acc: 0.949')
	# print('batch_size: 64 - val_acc: 0.940')
	# print('batch_size: 128 - val_acc: 0.938')
