import glob

from PIL import Image


images = glob.glob('aerial_images_128/acoperis/*.png')
masks = glob.glob('aerial_images_128/acoperis_masks/*.png')
i = 1

for image in images:
	img_orig = Image.open(f'{image}')
	img_orig.transpose(Image.FLIP_LEFT_RIGHT).save(f'aerial_images_128/acoperis/img_{i}lr.png')
	img_orig.transpose(Image.FLIP_TOP_BOTTOM).save(f'aerial_images_128/acoperis/img_{i}tb.png')
	img_orig.transpose(Image.ROTATE_90).save(f'aerial_images_128/acoperis/img_{i}r90.png')
	img_orig.transpose(Image.ROTATE_180).save(f'aerial_images_128/acoperis/img_{i}r180.png')
	img_orig.transpose(Image.ROTATE_270).save(f'aerial_images_128/acoperis/img_{i}r270.png')
	i += 1

i = 1

for mask in masks:
	mask_orig = Image.open(f'{mask}')
	mask_orig.transpose(Image.FLIP_LEFT_RIGHT).save(f'aerial_images_128/acoperis_masks/img_{i}lr.png')
	mask_orig.transpose(Image.FLIP_TOP_BOTTOM).save(f'aerial_images_128/acoperis_masks/img_{i}tb.png')
	mask_orig.transpose(Image.ROTATE_90).save(f'aerial_images_128/acoperis_masks/img_{i}r90.png')
	mask_orig.transpose(Image.ROTATE_180).save(f'aerial_images_128/acoperis_masks/img_{i}r180.png')
	mask_orig.transpose(Image.ROTATE_270).save(f'aerial_images_128/acoperis_masks/img_{i}r270.png')
	i += 1


# images = glob.glob('aerial_images_256/acoperis/*.png')
# masks = glob.glob('aerial_images_256/acoperis_masks/*.png')
# i = 1
#
# for image in images:
# 	img_orig = Image.open(f'{image}')
# 	img_orig.transpose(Image.FLIP_LEFT_RIGHT).save(f'aerial_images_256/acoperis/img_{i}lr.png')
# 	img_orig.transpose(Image.FLIP_TOP_BOTTOM).save(f'aerial_images_256/acoperis/img_{i}tb.png')
# 	img_orig.transpose(Image.ROTATE_90).save(f'aerial_images_256/acoperis/img_{i}r90.png')
# 	img_orig.transpose(Image.ROTATE_180).save(f'aerial_images_256/acoperis/img_{i}r180.png')
# 	img_orig.transpose(Image.ROTATE_270).save(f'aerial_images_256/acoperis/img_{i}r270.png')
# 	i += 1
#
# i = 1
#
# for mask in masks:
# 	mask_orig = Image.open(f'{mask}')
# 	mask_orig.transpose(Image.FLIP_LEFT_RIGHT).save(f'aerial_images_256/acoperis_masks/img_{i}lr.png')
# 	mask_orig.transpose(Image.FLIP_TOP_BOTTOM).save(f'aerial_images_256/acoperis_masks/img_{i}tb.png')
# 	mask_orig.transpose(Image.ROTATE_90).save(f'aerial_images_256/acoperis_masks/img_{i}r90.png')
# 	mask_orig.transpose(Image.ROTATE_180).save(f'aerial_images_256/acoperis_masks/img_{i}r180.png')
# 	mask_orig.transpose(Image.ROTATE_270).save(f'aerial_images_256/acoperis_masks/img_{i}r270.png')
# 	i += 1
