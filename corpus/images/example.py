import sklearn
from sklearn import mixture
from sklearn import cluster
from .metric import Metric
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Example:
    def __init__(self, example_path, masks_present=True):
        self.example_path = example_path
        image_dir = os.path.join(example_path, 'images')
        images = os.listdir(image_dir)
        images = [os.path.join(image_dir, image) for image in images]

        self.image_path = images[0]
        self.image_id = self.image_path.split('/')[-1].split('.')[0]
        self.image = mpimg.imread(self.image_path)
        self.image_shape = self.image.shape
        self.gray_image = self.rgb2gray(self.image)
        if masks_present:
            mask_dir = os.path.join(example_path, 'masks')
            masks = os.listdir(mask_dir)
            masks = [os.path.join(mask_dir, mask) for mask in masks]
            self.mask_paths = masks
            self.mask = self.combine_masks(self.mask_paths)

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
   
    def combine_masks(self, masks):
        res = np.sum(np.array([ mpimg.imread(mask) for mask in masks]), axis=0)
        res[res != 0] = 1
        return res
   
    def nuclei_vals(self):
        flat_img = self.gray_image.reshape(-1)
        flat_msk = self.combined_mask.reshape(-1).astype('bool')
        return flat_img[flat_msk], flat_img[~flat_msk]

    def set_prediction(self, predictions):
        assert predictions.shape == self.gray_image.shape
        self.predictions = predictions

    def get_csv_line(self):
        return self.image_id + ',' + self.rle_encoding()
       
    def rle_encoding(self):
        '''
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        Returns run length as list
        https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
        '''
        x = self.predictions
        dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b+1, 0))
            run_lengths[-1] += 1
            prev = b
        return " ".join([str(el) for el in run_lengths])

    def compute_metric(self):
        self.m = Metric(self.predictions, self.mask)
        self.m.compute_metric()


