from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf


def conv_block(inputs, filters, pool=True):
	x = Conv2D(filters, 3, padding='same')(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(filters, 3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	if pool == True:
		p = MaxPool2D((2, 2))(x)

		return x, p
	else:
		return x


def build_unet(shape, num_classes):
	inputs = Input(shape)

	# Encoder
	x1, p1 = conv_block(inputs, 32, pool=True)
	x2, p2 = conv_block(p1, 64, pool=True)
	x3, p3 = conv_block(p2, 96, pool=True)
	x4, p4 = conv_block(p3, 128, pool=True)

	# Bridge
	b1 = conv_block(p4, 256, pool=False)

	# Decoder
	u1 = UpSampling2D((2, 2), interpolation='bilinear')(b1)
	c1 = Concatenate()([u1, x4])
	x5 = conv_block(c1, 128, pool=False)

	u2 = UpSampling2D((2, 2), interpolation='bilinear')(x5)
	c2 = Concatenate()([u2, x3])
	x6 = conv_block(c2, 96, pool=False)

	u3 = UpSampling2D((2, 2), interpolation='bilinear')(x6)
	c3 = Concatenate()([u3, x2])
	x7 = conv_block(c3, 64, pool=False)

	u4 = UpSampling2D((2, 2), interpolation='bilinear')(x7)
	c4 = Concatenate()([u4, x1])
	x8 = conv_block(c4, 32, pool=False)

	# Output layer
	output = Conv2D(num_classes, 1, padding='same', activation='softmax')(x8)

	return Model(inputs, output)


def get_unet(input_shape):
	#https://github.com/ShawDa/unet-rgb/blob/master/unet.py
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


def get_efficientnet(input_shape):
	# https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
	# Stem
	def stem(input):
		# input layer
		si = Input(input, filters, kernel_size)
		# rescale
		sr = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(si)
		# normalization
		sn = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(sr)
		# zero padding
		szp = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(sn)
		# conv2d
		sc = tf.keras.layers.Conv2D(filters, kernel_size)(szp)
		# batch normalization
		sb = tf.keras.layers.BatchNormalization()(sc)
		# activation
		sa = tf.keras.layers.ReLU()(sb)

		return sa

	# Module 1
	def module1(input, kernel_size):
		# depthconv2d
		m1d = tf.keras.layers.DepthwiseConv2D(kernel_size)(input)
		# batch normalization
		m1b = tf.keras.layers.BatchNormalization()(m1d)
		# activation
		m1a = tf.keras.layers.ReLU()(m1b)

		return m1a

	# Module 2
	def module2(input, kernel_size):
		# depthconv2d
		m2d = tf.keras.layers.DepthwiseConv2D(kernel_size)(input)
		# batch normalization
		m2b = tf.keras.layers.BatchNormalization()(m2d)
		# activation
		m2a = tf.keras.layers.ReLU()(m2b)
		# zero padding
		m2zp = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(m2a)
		# depthconv2d
		m2d = tf.keras.layers.DepthwiseConv2D(kernel_size)(m2zp)
		# batch normalization
		m2b = tf.keras.layers.BatchNormalization()(m2d)
		# activation
		m2a = tf.keras.layers.ReLU()(m2b)

		return m2a

	# Module 3
	def module3(input, filters, kernel_size):
		# global averrage pooling
		m3gap = tf.keras.layers.GlobalAveragePolling2D()(input)
		# rescale
		m3r = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(m3gap)
		# Conv2D
		m3c = tf.keras.layers.Conv2D(filters, kernel_size)(m3r)
		# Conv2D
		m3c = tf.keras.layers.Conv2D(filters, kernel_size)(m3c)

		return m3c

	# Final layer
	def final(input, filters, kernel_size):
		# Conv2D
		fc = tf.keras.layers.Conv2D(filters, kernel_size)(m3c)
		# batch normalization
		fb = tf.keras.layers.BatchNormalization()(fc)
		# activation
		fa = tf.keras.layers.ReLU()(fb)

		return fb

	# Order
	# Stem
	stem = stem(input_shape, 32, 3)
	# M1 - block 1
	b1 = module1(stem, 3)
	# M2, M3, Add - block 2
	m2 = module2(b1, 3)
	m3 = module3(m2, 24, 3)
	b2 = tf.keras.layers.Add()([m2, m3])
	# M2, M3, Add - block 3
	m2 = module2(b2, 5)
	m3 = module3(m2, 40, 5)
	b3 = tf.keras.layers.Add()([m2, m3])
	# M2, M3, Add, M3, Add - block 4
	m2 = module2(b3, 3)
	m3 = module3(m2, 80, 3)
	a4 = tf.keras.layers.Add()([m2, m3])
	m3 = module3(a4, 80, 3)
	b4 = tf.keras.layers.Add()([a4, m3])
	# M2, M3, Add, M3, Add - block 5
	m2 = module2(b4, 5)
	m3 = module3(m2, 112, 5)
	a5 = tf.keras.layers.Add()([m2, m3])
	m3 = module3(a5, 112, 5)
	b5 = tf.keras.layers.Add()([a5, m3])
	# M2, M3, Add, M3, Add, M3, Add - block 6
	m2 = module2(b5, 5)
	m3 = module3(m2, 192, 5)
	a6 = tf.keras.layers.Add()([m2, m3])
	m3 = module3(a6, 192, 5)
	a6 = tf.keras.layers.Add()([a6, m3])
	m3 = module3(a6, 192, 5)
	b6 = tf.keras.layers.Add()([a6, m3])
	# M2 - block 7
	b7 = module2(b6, 3)
	# Final layer
	f = final(b7, 1280, 3)

	return f


if __name__ == '__main__':
	model = build_unet((320, 512, 3), 10)
	model.summary()
