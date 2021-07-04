from PIL import Image
from time import time

import glob
import numpy as np


def crop_image(im, size):
	image_width, image_height = im.size

	for i in range(image_height // size):
		for j in range(image_width // size):
			box = (j * size, i * size, (j + 1) * size, (i + 1) * size)

			yield im.crop(box), j * size, i * size


if __name__ == '__main__':
	start_time = time()
	# train_directory = glob.glob('../../../Desktop/AerialImageDataset/train/images/*.tif')
	# test_directory_images = glob.glob('../../../Desktop/AerialImageDataset/test/images/*.tif')
	test_directory_gt = glob.glob('../../../Desktop/AerialImageDataset/test/gt/*.tif')
	size = 1024
	index = 0

	for infile in test_directory_gt:
		filename = infile.split('/')[-1].split('.')[0]
		im = Image.open(infile)
		start_num = 0

		for k, (piece, a, b) in enumerate(crop_image(im, size), start_num):
			index += 1
			# folder = m_histograma(b, a)
			img = Image.new('RGB', (size, size), 255)
			img.paste(piece)
			img.save(f'big/aerial_image_dataset_1024/validation/gt/{filename}{index}.png')

	print(f'Ready: {round(time() - start_time, 2)}')
