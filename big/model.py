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


def get_unet_efficientnet(input_shape):
	# https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
	i_s = Input(input_shape)

	def stem(input, filters, kernel_size):
		# input layer, rescale, normalization, zero padding, convolution, batch normalization, activation
		sr = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(input)
		sn = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(sr)
		szp = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(sn)
		sc = tf.keras.layers.Conv2D(filters, kernel_size)(szp)
		sb = tf.keras.layers.BatchNormalization()(sc)
		sa = tf.keras.layers.ReLU()(sb)

		return sa

	def module1(input, kernel_size):
		# depthconv2d, batch normalization, activation
		m1d = tf.keras.layers.DepthwiseConv2D(kernel_size)(input)
		m1b = tf.keras.layers.BatchNormalization()(m1d)
		m1a = tf.keras.layers.ReLU()(m1b)

		return m1a

	def module2(input, kernel_size):
		# depthconv2d, batch normalization, activation, zero padding, depthconv2d, batch normalization, activation
		m2d = tf.keras.layers.DepthwiseConv2D(kernel_size)(input)
		m2b = tf.keras.layers.BatchNormalization()(m2d)
		m2a = tf.keras.layers.ReLU()(m2b)
		m2zp = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(m2a)
		m2d = tf.keras.layers.DepthwiseConv2D(kernel_size)(m2zp)
		m2b = tf.keras.layers.BatchNormalization()(m2d)
		m2a = tf.keras.layers.ReLU()(m2b)

		return m2a

	def module3(input, filters, kernel_size):
		# global averrage pooling, rescale, convolution, convolution
		# m3gap = tf.keras.layers.GlobalAveragePooling2D()(input)
		# print(m3gap.shape)
		m3r = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(input)
		m3c = tf.keras.layers.Conv2D(filters, kernel_size)(m3r)
		m3c = tf.keras.layers.Conv2D(filters, kernel_size)(m3c)

		m3inputc = tf.keras.layers.Conv2D(filters, kernel_size)(input)
		m3inputc = tf.keras.layers.Conv2D(filters, kernel_size)(m3inputc)
		m3a = tf.keras.layers.Add()([m3c, m3inputc])

		return m3c

	def final_layer(input, filters, kernel_size):
		# convolution, batch normalization, activation
		fc = tf.keras.layers.Conv2D(filters, kernel_size)(input)
		fb = tf.keras.layers.BatchNormalization()(fc)
		fa = tf.keras.layers.ReLU()(fb)

		return fb

	# Stem
	stem = stem(i_s, 32, 3)

	# Block 1 - M1
	b1 = module1(stem, 3)

	# Bblock 2 - M2, M3, Add
	b2_m2 = module2(b1, 3)
	b2 = module3(b2_m2, 24, 3)

	# Block 3 - M2, M3, Add
	b3_m2 = module2(b2, 5)
	b3 = module3(b3_m2, 40, 5)

	# Block 4 - M2, M3, Add, M3, Add
	b4_m2 = module2(b3, 3)
	b4_m3 = module3(b4_m2, 80, 3)
	b4 = module3(b4_m3, 80, 3)

	# Block 5 - M2, M3, Add, M3, Add
	b5_m2 = module2(b4, 5)
	b5_m3 = module3(b5_m2, 112, 5)
	b5 = module3(b5_m3, 112, 5)

	# Block 6 - M2, M3, Add, M3, Add, M3, Add
	b6_m2 = module2(b5, 5)
	b6_m3 = module3(b6_m2, 192, 5)
	b6_m3 = module3(b6_m3, 192, 5)
	b6 = module3(b6_m3, 192, 5)

	# Block 7 - M2
	b7 = module2(b6, 3)

	# Final layer
	f = final_layer(b7, 1280, 3)

	# conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	# conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	#
	# conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	# conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	# pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	#
	# conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	# conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	# pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	#
	# conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	# conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	# drop4 = Dropout(0.5)(conv4)
	# pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
	#
	# conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	# conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	# drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(f))
	# merge6 = Concatenate(axis=3) ([drop4, up6])
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
	# merge7 = Concatenate(axis=3) ([conv3, up7])
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
	# merge8 = Concatenate(axis=3) ([conv2, up8])
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
	# merge9 = Concatenate(axis=3) ([conv1, up9])
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

	# output = Conv2D(1, 1, activation='sigmoid')(conv9)
	output = Conv2D(1, 1, activation='softmax')(conv9)

	return Model(i_s, output)


if __name__ == '__main__':
	model = build_unet((320, 512, 3), 10)
	model.summary()
