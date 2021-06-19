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


def get_efficientnet_unet(input_shape):
	# Efficientnet b0 as contracting path for unet
	# https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
	# https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-3-mbconv-squeeze-and-excitation-and-more-4ca9fd62d302
	i_s = Input(input_shape)

	# class ConvBnAct(nn.Module):
	#   """Layer grouping a convolution, batchnorm, and activation function"""
	#   def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=0, groups=1, bias=False, bn=True, act=True):
	#     self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
	#     self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
	#     self.act = nn.SiLU() if act else nn.Identity()


	def c_bn_a(input_shape, filters, kernel_size=3, stride=1, padding=0, groups=1, bias=False, act=True):
		# convolution, batch_normalization, activation
		c2d = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, stride=stride, padding=padding)(input_shape)
		bn = tf.keras.layers.BatchNormalization()(c2d)
		a = tf.keras.layers.SiLU()(bn) if act else tf.identity()(bn)

		return a

	# class SEBlock(nn.Module):
	#   """Squeeze-and-excitation block"""
	#   def __init__(self, n_in, r=24):
	#     self.squeeze = nn.AdaptiveAvgPool2d(1)
	#     self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
	#                                     nn.SiLU(),
	#                                     nn.Conv2d(n_in//r, n_in, kernel_size=1),
	#                                     nn.Sigmoid())


	def s_e_(input_shape, filters, r=24):
		# For other output sizes in Keras, you need to use AveragePooling2D, but you can't specify the output shape directly. You need to calculate/define the pool_size, stride, and padding parameters depending on how you want the output shape. If you need help with the calculations, check this page of CS231n course.
		# https://cs231n.github.io/convolutional-networks/#pool
		ap2d = tf.keras.layers.AverragePooling2D()(input_shape)
		c2d1 = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(input_shape)
		silu = tf.keras.layers.SiLU()(c2d1)
		c2d2 = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(silu)
		sig = tf.keras.layers.Sigmoid()(c2d2)

		return input_shape * sig

	# class DropSample(nn.Module):
	#   """Drops each sample in x with probability p during training"""
	#   def __init__(self, p=0):
	#   def forward(self, x):
	#     if (not self.p) or (not self.training):
	#       return x
	#
	#     batch_size = len(x)
	#     random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
	#     bit_mask = self.p<random_tensor
	#
	#     x = x.div(1-self.p)
	#     x = x * bit_mask
	#     return x

	def drop_sample(input_shape, p=0):
		# Drops each sample in x with probability p during training
		batch_size = len(input_shape)
		# random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
		bit_mask = p < random_tensor

		input_shape = input_shape.div(1 - p)
		input_shape *= bit_mask

		return input_shape

	# class MBConvN(nn.Module):
	#   """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
	#   def __init__(self, n_in, n_out, expansion_factor, kernel_size=3, stride=1, r=24, p=0):
	#     super().__init__()
	#
	#     padding = (kernel_size-1)//2
	#     expanded = expansion_factor*n_in
	#     self.skip_connection = (n_in == n_out) and (stride == 1)
	#
	#     self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded, kernel_size=1)
	#     self.depthwise = ConvBnAct(expanded, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=expanded)
	#     self.se = SEBlock(expanded, r=r)
	#     self.reduce_pw = ConvBnAct(expanded, n_out, kernel_size=1, act=False)
	#     self.dropsample = DropSample(p)
	#
	#   def forward(self, x):
	#     residual = x
	#
	#     x = self.expand_pw(x)
	#     x = self.depthwise(x)
	#     x = self.se(x)
	#     x = self.reduce_pw(x)
	#
	#     if self.skip_connection:
	#       x = self.dropsample(x)
	#       x = x + residual
	#
	#     return x

	def mb_conv_n(input_shape, filters, expansion_factor=1, kernel_size=3, stride=1, r=24, p=0):
		# MBConv with an expansion factor of N, plus squeeze-and-excitation
		padding = (kernel_size - 1) // 2
		expanded = expansion_factor * input_shape
		skip_connection = (input_shape == filters) and (stride == 1)

		expand_pw = tf.identity(input_shape) if (expansion_factor == 1) else c_bn_a(input_shape, expanded, kernel_size=1)
		depthwise = c_bn_a(expand_pw, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=expanded)
		se = s_e(depthwise, filters, r=r)
		reduce_pw = c_bn_a(se, filters, kernel_size=1, act=False)
		# drop_sample = drop_sample(p)

		return reduce_pw

	# expansion_factor=6

	# Block 1
	bl1 = tf.keras.layers.Conv2D(32, kernel_size=3)(i_s)

	# Block 2
	bl2 = mb_conv_n(bl1, 16)

	# Block 3
	bl3 = mb_conv_n(bl2, 24, expansion_factor=6)

	# Block 4
	bl4 = mb_conv_n(bl3, 40, expansion_factor=6, kernel_size=5)

	# Block 5
	bl5 = mb_conv_n(bl4, 80, expansion_factor=6)

	# Block 6
	bl6 = mb_conv_n(bl5, 112, expansion_factor=6, kernel_size=5)

	# Block 7
	bl7 = mb_conv_n(bl6, 192, expansion_factor=6, kernel_size=5)

	# Block 8
	bl8 = mb_conv_n(bl7, 320, expansion_factor=6)

	# Block 9
	c9 = tf.keras.layers.Conv2D(1280, kernel_size=1)(bl8)
	p9 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(c9)
	# p9 = tf.keras.layers.GlobalAveragePooling2D()(c9)
	bl9 = tf.keras.layers.Dense(units)(p9)

	# def fc(x, num_units_out, name, seed=None):
			# with tf.variable_scope(name, use_resource=True):
				# x = tf.keras.layers.Dense(inputs=x, units=num_units_out, kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

	# Unet expanding path
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
	model = build_unet((320, 512, 3), 10)
	model.summary()
