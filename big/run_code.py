import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model_s import *


if __name__ == '__main__':
	# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
	# config.gpu_options.allow_growth = True
	# session = tf.compat.v1.Session(config=config)
	print('predict test data')
	# model = get_unet((256, 256, 3))
	# model.load_weights('unet.hdf5')

	model = get_efficientnet_unet((256, 256, 3))
	model.load_weights('efficientnet.hdf5')

	# model = get_efficientnet_as_unet((256, 256, 3))
	# model.load_weights('efficientnet_as_unet.hdf5')

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

	# mask = np.floor(imgs_mask_test * 255)

	imgs_mask_test[imgs_mask_test <= 0.5] = 0
	imgs_mask_test[imgs_mask_test > 0.5] = 1

	plt.figure(2)
	plt.imshow(imgs_mask_test.astype(dtype=np.uint8), vmin=0, vmax=1)

	for i in range(256):
		for j in range(256):
			if imgs_mask_test[i, j] != 1:
				img_test[i, j, :] = 0

	plt.figure(3)
	plt.imshow(img_test)
	plt.show()
