from PIL import Image
import numpy as np
import os

data_path_0 = 'DATA/0'
data_path_1 = 'DATA/1'

names_0 = os.listdir(data_path_0)
names_1 = os.listdir(data_path_1)

#print(len(names_0))
#print(len(names_1))

def convert_to_grayscale(image_path):
    Image.open(image_path).convert('L').save(image_path)


rnd_idx = np.random.randint(0, len(names_0), size=1000)
#print(rnd_idx)

for i in range(0, len(rnd_idx)):
	im = Image.open(os.path.join(data_path_0, f'{names_0[rnd_idx[i]]}'))
	crop_rectangle = (100, 100, 160, 160)
	cropped_im = im.crop(crop_rectangle)
	cropped_im = cropped_im.save(os.path.join('NEW_DATA/0', f'{names_0[rnd_idx[i]]}'))
	convert_to_grayscale(os.path.join('NEW_DATA/0', f'{names_0[rnd_idx[i]]}'))


rnd_idx = np.random.randint(0, len(names_1), size=1000)

for i in range(0, len(rnd_idx)):
	im = Image.open(os.path.join(data_path_1, f'{names_1[rnd_idx[i]]}'))
	crop_rectangle = (100, 100, 160, 160)
	cropped_im = im.crop(crop_rectangle)
	cropped_im = cropped_im.save(os.path.join('NEW_DATA/1', f'{names_1[rnd_idx[i]]}'))
	convert_to_grayscale(os.path.join('NEW_DATA/1', f'{names_1[rnd_idx[i]]}'))
