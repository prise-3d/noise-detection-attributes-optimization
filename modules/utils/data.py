from ipfml import processing, metrics, utils

from modules.utils.config import *
from modules.utils.filters import w2d

import cv2

from PIL import Image
from skimage import color
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd as lin_svd

from scipy.signal import medfilt2d, wiener, cwt

import numpy as np


_scenes_names_prefix   = '_scenes_names'
_scenes_indices_prefix = '_scenes_indices'

# store all variables from current module context
context_vars = vars()


def get_svd_data(data_type, block):
    """
    Method which returns the data type expected
    """

    if 'filters_statistics' in data_type:

        img_width, img_height = 200, 200

        lab_img = metrics.get_LAB_L(block)
        arr = np.array(lab_img)

        # compute all filters statistics
        def get_stats(arr, I_filter):

            e1       = np.abs(arr - I_filter)
            L        = np.array(e1)
            mu0      = np.mean(L)
            A        = L - mu0
            H        = A * A
            E        = np.sum(H) / (img_width * img_height)
            P        = np.sqrt(E)

            return mu0, P

        stats = []

        kernel = np.ones((3,3),np.float32)/9
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        kernel = np.ones((5,5),np.float32)/25
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1.5)))

        stats.append(get_stats(arr, medfilt2d(arr, [3, 3])))

        stats.append(get_stats(arr, medfilt2d(arr, [5, 5])))

        stats.append(get_stats(arr, wiener(arr, [3, 3])))

        stats.append(get_stats(arr, wiener(arr, [5, 5])))

        wave = w2d(arr, 'db1', 2)
        stats.append(get_stats(arr, np.array(wave, 'float64')))

        data = []

        for stat in stats:
            data.append(stat[0])

        for stat in stats:
            data.append(stat[1])
        
        data = np.array(data)

    return data


def get_highest_values(arr, n):
    return np.array(arr).argsort()[-n:][::-1]


def get_lowest_values(arr, n):
    return np.array(arr).argsort()[::-1][-n:][::-1]


def _get_mscn_variance(block, sub_block_size=(50, 50)):

    blocks = processing.divide_in_blocks(block, sub_block_size)

    data = []

    for block in blocks:
        mscn_coefficients = processing.get_mscn_coefficients(block)
        flat_coeff = mscn_coefficients.flatten()
        data.append(np.var(flat_coeff))

    return np.sort(data)


def get_renderer_scenes_indices(renderer_name):

    if renderer_name not in renderer_choices:
        raise ValueError("Unknown renderer name")

    if renderer_name == 'all':
        return scenes_indices
    else:
        return context_vars[renderer_name + _scenes_indices_prefix]

def get_renderer_scenes_names(renderer_name):

    if renderer_name not in renderer_choices:
        raise ValueError("Unknown renderer name")

    if renderer_name == 'all':
        return scenes_names
    else:
        return context_vars[renderer_name + _scenes_names_prefix]

