import os
import numpy as np
from PIL import Image


MAXINTENS = 255


def save_img_data(data, name):
    """stores previously generated img data"""
    if not os.path.isdir(os.path.dirname(name)) and os.path.dirname(name) != '':
        os.makedirs(os.path.dirname(name))
    data.save(name)


def visualize_bw(img, name):
    """visualizes a single black and white img (2D numpy array)"""
    if img.dtype == bool:
        img = img.astype(np.uint8) * MAXINTENS
    else:
        if img.max() == img.min():
            img = np.zeros(img.shape, dtype=np.uint8)
        else:
            img = (np.divide(img.astype('double') - img.min(), img.max() - img.min()) * MAXINTENS).astype(np.uint8) # normalize to 0-255
    bw_data = Image.fromarray(img, mode='L')
    save_img_data(bw_data, name)


def visualize_rgb(img, name, bands, min_v, max_v):
    """visualizes a single multicolor img (3D numpy array) with specified bands as RGB channels"""
    assert(max(bands) < img.shape[2])
    img = img[:, :, bands]
    for dim in range(img.shape[2]):
        img[:, :, dim] = np.clip(img[:, :, dim], min_v[dim], max_v[dim]) # clip to min max values
    img = (np.divide(img.astype(float) - min_v, max_v - min_v) * MAXINTENS).astype(np.uint8) # normalize to 0-255
    rgb_data = Image.fromarray(img, mode='RGB')
    save_img_data(rgb_data, name)


class Visualizer:
    """Creates images of a dataset, based on minimum and maximum color values and three specific bands that are chosen for RGB visualization"""
    def __init__(self, minmax, bands):
        if bands is None:
            self.bands = np.arange(3)
        else:
            self.bands = np.array(bands)
        assert(len(self.bands) == 3)
        assert(len(minmax[0]) == len(minmax[1]))
        assert(len(minmax[0]) == len(self.bands)) # min max values for each band
        assert(min(self.bands) >= 0)
        self.min_v = minmax[0]
        self.max_v = minmax[1]


    def visualize_all(self, imgs, out, fname='img'):
        """visualizes a a bunch of RGB or black and white images (4D or 3D numpy array)"""
        if len(imgs.shape) == 2:
            np.expand_dims(imgs, axis=0)
        for t in range(imgs.shape[0]):
            if len(imgs.shape) == 4: # RGB data
                visualize_rgb(imgs[t, :, :, :], '{}/{}_t{:03d}.png'.format(out, fname, t), self.bands, self.min_v, self.max_v)
            else: # BW data
                visualize_bw(imgs[t, :, :], '{}/{}_t{:03d}.png'.format(out, fname, t))
                
