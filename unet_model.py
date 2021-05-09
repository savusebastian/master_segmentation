from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D
from tensorflow.keras.models import Model


# Function to define the UNET Model
def get_unet(input_img, n_filters=8, dropout=0.2):
	# Contracting Path
	c1 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(input_img)
	c1 = BatchNormalization()(c1)
	c1 = Activation('relu')(c1)
	p1 = MaxPooling2D((2, 2))(c1)
	# p1 = Dropout(dropout)(p1)

	c2 = Conv2D(n_filters * 2, 3, kernel_initializer='he_normal', padding='same')(p1)
	c2 = BatchNormalization()(c2)
	c2 = Activation('relu')(c2)
	p2 = MaxPooling2D((2, 2))(c2)
	# p2 = Dropout(dropout)(p2)

	c3 = Conv2D(n_filters * 4, 3, kernel_initializer='he_normal', padding='same')(p2)
	c3 = BatchNormalization()(c3)
	c3 = Activation('relu')(c3)
	p3 = MaxPooling2D((2, 2))(c3)
	# p3 = Dropout(dropout)(p3)

	c4 = Conv2D(n_filters * 8, 3, kernel_initializer='he_normal', padding='same')(p3)
	c4 = BatchNormalization()(c4)
	c4 = Activation('relu')(c4)
	p4 = MaxPooling2D((2, 2))(c4)
	# p4 = Dropout(dropout)(p4)

	c5 = Conv2D(n_filters * 16, 3, kernel_initializer='he_normal', padding='same')(p4)
	c5 = BatchNormalization()(c5)
	c5 = Activation('relu')(c5)

	# Expansive Path
	u6 = Conv2DTranspose(n_filters * 8, 3, strides=(2, 2), padding='same')(c5)
	u6 = concatenate([u6, c4])
	# u6 = Dropout(dropout)(u6)
	c6 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u6)
	c6 = BatchNormalization()(c6)
	c6 = Activation('relu')(c6)

	u7 = Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding='same')(c6)
	u7 = concatenate([u7, c3])
	# u7 = Dropout(dropout)(u7)
	c7 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u7)
	c7 = BatchNormalization()(c7)
	c7 = Activation('relu')(c7)

	u8 = Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding='same')(c7)
	u8 = concatenate([u8, c2])
	# u8 = Dropout(dropout)(u8)
	c8 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u8)
	c8 = BatchNormalization()(c8)
	c8 = Activation('relu')(c8)

	u9 = Conv2DTranspose(n_filters * 1, 3, strides=(2, 2), padding='same')(c8)
	u9 = concatenate([u9, c1])
	# u9 = Dropout(dropout)(u9)
	c9 = Conv2D(n_filters, 3, kernel_initializer='he_normal', padding='same')(u9)
	c9 = BatchNormalization()(c9)
	c9 = Activation('relu')(c9)

	outputs = Conv2D(1, 1, activation='sigmoid')(c9)
	model = Model(inputs=[input_img], outputs=[outputs])

	return model
