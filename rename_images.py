import glob
import os


images = glob.glob('./aerial_images_128/acoperis/*.png')
masks = glob.glob('./aerial_images_128/acoperis_masks/*.png')
i = 1

for image in images:
	os.rename(image, f'./aerial_images_128/acoperis/a{i}.png')
	i += 1

i = 1

for mask in masks:
	os.rename(mask, f'./aerial_images_128/acoperis_masks/a{i}.png')
	i += 1


images = glob.glob('./aerial_images_128/acoperis/*.png')
masks = glob.glob('./aerial_images_128/acoperis_masks/*.png')
i = 1

for image in images:
	os.rename(image, f'./aerial_images_128/acoperis/img_{i}.png')
	i += 1

i = 1

for mask in masks:
	os.rename(mask, f'./aerial_images_128/acoperis_masks/img_{i}.png')
	i += 1


# images = glob.glob('./aerial_images_256/acoperis/*.png')
# masks = glob.glob('./aerial_images_256/acoperis_masks/*.png')
# i = 1
#
# for image in images:
# 	os.rename(image, f'./aerial_images_256/acoperis/a{i}.png')
# 	i += 1
#
# i = 1
#
# for mask in masks:
# 	os.rename(mask, f'./aerial_images_256/acoperis_masks/a{i}.png')
# 	i += 1
#
#
# images = glob.glob('./aerial_images_256/acoperis/*.png')
# masks = glob.glob('./aerial_images_256/acoperis_masks/*.png')
# i = 1
#
# for image in images:
# 	os.rename(image, f'./aerial_images_256/acoperis/img_{i}.png')
# 	i += 1
#
# i = 1
#
# for mask in masks:
# 	os.rename(mask, f'./aerial_images_256/acoperis_masks/img_{i}.png')
# 	i += 1
