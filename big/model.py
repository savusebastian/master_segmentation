from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
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


	def c_bn_a(input_shape, filters, kernel_size=3, act=True):
		# convolution, batch_normalization, activation
		c2d = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(input_shape)
		bn = tf.keras.layers.BatchNormalization()(c2d)
		a = tf.nn.silu(bn) if act else tf.identity(bn)

		return a

	# class SEBlock(nn.Module):
	#   """Squeeze-and-excitation block"""
	#   def __init__(self, n_in, r=24):
	#     self.squeeze = nn.AdaptiveAvgPool2d(1)
	#     self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
	#                                     nn.SiLU(),
	#                                     nn.Conv2d(n_in//r, n_in, kernel_size=1),
	#                                     nn.Sigmoid())


	def s_e(input_shape, filters, r=24):
		# For other output sizes in Keras, you need to use AveragePooling2D, but you can't specify the output shape directly. You need to calculate/define the pool_size, stride, and padding parameters depending on how you want the output shape. If you need help with the calculations, check this page of CS231n course.
		# https://cs231n.github.io/convolutional-networks/#pool
		ap2d = tf.keras.layers.AveragePooling2D()(input_shape)
		c2d1 = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(input_shape)
		a1 = tf.nn.silu(c2d1)
		c2d2 = tf.keras.layers.Conv2D(filters // r, kernel_size=1)(a1)
		a2 = tf.keras.activations.sigmoid(c2d2)

		return input_shape * a2

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

	# def drop_sample(input_shape, p=0):
	# 	# Drops each sample in x with probability p during training
	# 	batch_size = input_shape.shape
	# 	# random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
	# 	bit_mask = p < random_tensor
	#
	# 	input_shape = input_shape.div(1 - p)
	# 	input_shape *= bit_mask
	#
	# 	return input_shape

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

	def mb_conv_n(input_shape, filters, expansion_factor=1, kernel_size=3, r=24, p=0):
		# MBConv with an expansion factor of N, plus squeeze-and-excitation
		padding = (kernel_size - 1) // 2
		expanded = expansion_factor * input_shape.shape[3]
		# skip_connection = (input_shape == filters) and (stride == 1)

		expand_pw = tf.identity(input_shape) if (expansion_factor == 1) else c_bn_a(input_shape, expanded, kernel_size=1)
		depthwise = c_bn_a(expand_pw, expanded, kernel_size=kernel_size)
		se = s_e(depthwise, filters, r=r)
		reduce_pw = c_bn_a(se, filters, kernel_size=1, act=False)
		# drop_sample = drop_sample(p)

		return reduce_pw


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
	bl9 = tf.keras.layers.Dense(1280)(p9)

	# def fc(x, num_units_out, name, seed=None):
			# with tf.variable_scope(name, use_resource=True):
				# x = tf.keras.layers.Dense(inputs=x, units=num_units_out, kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

	# Unet expanding path
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bl9)
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


# def get_efficientnet_as_unet(input_shape):
def EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate=0.2, drop_connect_rate=0.2, depth_divisor=8, activation='swish', blocks_args='default', model_name='efficientnet', include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=2, classifier_activation='softmax'):
	def block(inputs, activation='swish', drop_rate=0., name='', filters_in=32, filters_out=16, kernel_size=3, strides=1, expand_ratio=1, se_ratio=0., id_skip=True):
		# An inverted residual block.
		# Args:
		# 	inputs: input tensor.
		# 	activation: activation function.
		# 	drop_rate: float between 0 and 1, fraction of the input units to drop.
		# 	name: string, block label.
		# 	filters_in: integer, the number of input filters.
		# 	filters_out: integer, the number of output filters.
		# 	kernel_size: integer, the dimension of the convolution window.
		# 	strides: integer, the stride of the convolution.
		# 	expand_ratio: integer, scaling coefficient for the input filters.
		# 	se_ratio: float between 0 and 1, fraction to squeeze the input filters.
		# 	id_skip: boolean.
		# Returns:
		# 	output tensor for the block.

		bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

		# Expansion phase
		filters = filters_in * expand_ratio
		if expand_ratio != 1:
			x = layers.Conv2D(filters, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'expand_conv')(inputs)
			x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
			x = layers.Activation(activation, name=name + 'expand_activation')(x)
		else:
			x = inputs

		# Depthwise Convolution
		if strides == 2:
			x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size), name=name + 'dwconv_pad')(x)
			conv_pad = 'valid'
		else:
			conv_pad = 'same'

		x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + 'dwconv')(x)
		x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
		x = layers.Activation(activation, name=name + 'activation')(x)

		# Squeeze and Excitation phase
		if 0 < se_ratio <= 1:
			filters_se = max(1, int(filters_in * se_ratio))
			se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)

			if bn_axis == 1:
				se_shape = (filters, 1, 1)
			else:
				se_shape = (1, 1, filters)

			se = layers.Reshape(se_shape, name=name + 'se_reshape')(se)
			se = layers.Conv2D(filters_se, 1, padding='same', activation=activation, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_reduce')(se)
			se = layers.Conv2D(filters, 1, padding='same', activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_expand')(se)
			x = layers.multiply([x, se], name=name + 'se_excite')

		# Output phase
		x = layers.Conv2D(filters_out, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'project_conv')(x)
		x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)

		if id_skip and strides == 1 and filters_in == filters_out:
			if drop_rate > 0:
				x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)

			x = layers.add([x, inputs], name=name + 'add')

		return x


	BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/keras-applications/'

	WEIGHTS_HASHES = {
		'b0': ('902e53a9f72be733fc0bcb005b3ebbac', '50bc09e76180e00e4465e1a485ddc09d'),
		'b1': ('1d254153d4ab51201f1646940f018540', '74c4e6b3e1f6a1eea24c589628592432'),
		'b2': ('b15cce36ff4dcbd00b6dd88e7857a6ad', '111f8e2ac8aa800a7a99e3239f7bfb39'),
		'b3': ('ffd1fdc53d0ce67064dc6a9c7960ede0', 'af6d107764bb5b1abb91932881670226'),
		'b4': ('18c95ad55216b8f92d7e70b3a046e2fc', 'ebc24e6d6c33eaebbd558eafbeedf1ba'),
		'b5': ('ace28f2a6363774853a83a0b21b9421a', '38879255a25d3c92d5e44e04ae6cec6f'),
		'b6': ('165f6e37dce68623721b423839de8be5', '9ecce42647a20130c1f39a5d4cb75743'),
		'b7': ('8c03f828fec3ef71311cd463b6759d99', 'cbcfe4450ddf6f3ad90b1b398090fe4a'),
	}

	DEFAULT_BLOCKS_ARGS = [
		{'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16, 'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
		{'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
		{'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
		{'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
		{'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112, 'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
		{'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192, 'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
		{'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320, 'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
	]

	CONV_KERNEL_INITIALIZER = {
		'class_name': 'VarianceScaling',
		'config': {
			'scale': 2.0,
			'mode': 'fan_out',
			'distribution': 'truncated_normal'
		}
	}

	DENSE_KERNEL_INITIALIZER = {
		'class_name': 'VarianceScaling',
		'config': {
			'scale': 1. / 3.,
			'mode': 'fan_out',
			'distribution': 'uniform'
		}
	}

	layers = VersionAwareLayers()

	if blocks_args == 'default':
		blocks_args = DEFAULT_BLOCKS_ARGS

	# if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
	# 	raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.')

	# if weights == 'imagenet' and include_top and classes != 1000:
	# 	raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000')

	# Determine proper input shape
	input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=default_size, min_size=32, data_format=backend.image_data_format(), require_flatten=include_top, weights=weights)

	if input_tensor is None:
		img_input = layers.Input(shape=input_shape)
	else:
		if not backend.is_keras_tensor(input_tensor):
			img_input = layers.Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

	def round_filters(filters, divisor=depth_divisor):
		# Round number of filters based on depth multiplier.
		filters *= width_coefficient
		new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
		# Make sure that round down does not go down by more than 10%.
		if new_filters < 0.9 * filters:
			new_filters += divisor

		return int(new_filters)

	def round_repeats(repeats):
		# Round number of repeats based on depth multiplier.
		return int(math.ceil(depth_coefficient * repeats))

	# Build stem
	x = img_input
	x = layers.Rescaling(1. / 255.)(x)
	x = layers.Normalization(axis=bn_axis)(x)

	x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, 3), name='stem_conv_pad')(x)
	x = layers.Conv2D(round_filters(32), 3, strides=2, padding='valid', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name='stem_conv')(x)
	x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
	x = layers.Activation(activation, name='stem_activation')(x)

	# Build blocks
	blocks_args = copy.deepcopy(blocks_args)

	b = 0
	blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))

	for (i, args) in enumerate(blocks_args):
		assert args['repeats'] > 0
		# Update block input and output filters based on depth multiplier.
		args['filters_in'] = round_filters(args['filters_in'])
		args['filters_out'] = round_filters(args['filters_out'])

		for j in range(round_repeats(args.pop('repeats'))):
			# The first block needs to take care of stride and filter size increase.
			if j > 0:
				args['strides'] = 1
				args['filters_in'] = args['filters_out']

			x = block(x, activation, drop_connect_rate * b / blocks, name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
			b += 1

	# Build top
	x = layers.Conv2D(round_filters(1280), 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name='top_conv')(x)
	x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
	x = layers.Activation(activation, name='top_activation')(x)

	if include_top:
		# x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

		if dropout_rate > 0:
			x = layers.Dropout(dropout_rate, name='top_dropout')(x)

		imagenet_utils.validate_activation(classifier_activation, weights)
		x = layers.Dense(classes, activation=classifier_activation, kernel_initializer=DENSE_KERNEL_INITIALIZER, name='predictions')(x)
	# else:
	# 	if pooling == 'avg':
	# 		x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
	# 	elif pooling == 'max':
	# 		x = layers.GlobalMaxPooling2D(name='max_pool')(x)

	# Ensure that the model takes into account any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = layer_utils.get_source_inputs(input_tensor)
	else:
		inputs = img_input

	# Create model.
	model = training.Model(inputs, x, name=model_name)

	# Load weights.
	# if weights == 'imagenet':
	# 	if include_top:
	# 		file_suffix = '.h5'
	# 		file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
	# 	else:
	# 		file_suffix = '_notop.h5'
	# 		file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
	# 		file_name = model_name + file_suffix
	# 		weights_path = data_utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash)
	# 		model.load_weights(weights_path)
	# elif weights is not None:
	# 	model.load_weights(weights)

	return model


@keras_export('keras.applications.efficientnet.EfficientNetB0', 'keras.applications.EfficientNetB0')
def EfficientNetB0(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=2, classifier_activation='softmax', **kwargs):
	return EfficientNet(1.0, 1.0, 256, 0.2, model_name='efficientnetb0', include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes, classifier_activation=classifier_activation, **kwargs)

@keras_export('keras.applications.efficientnet.preprocess_input')
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
	# A placeholder method for backward compatibility.
	# The preprocessing logic has been included in the efficientnet model
	# implementation. Users are no longer required to call this method to normalize
	# the input data. This method does nothing and only kept as a placeholder to
	# align the API surface between old and new version of model.
	# Args:
	#   x: A floating point `numpy.array` or a `tf.Tensor`.
	#   data_format: Optional data format of the image tensor/array. Defaults to
	#     None, in which case the global setting
	#     `tf.keras.backend.image_data_format()` is used (unless you changed it,
	#     it defaults to "channels_last").{mode}
	# Returns:
	#   Unchanged `numpy.array` or `tf.Tensor`.

	return x

@keras_export('keras.applications.efficientnet.decode_predictions')
def decode_predictions(preds, top=5):
	return imagenet_utils.decode_predictions(preds, top=top)


if __name__ == '__main__':
	model = build_unet((320, 512, 3), 10)
	model.summary()
