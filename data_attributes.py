# main imports
import numpy as np
import sys

# image transform imports
from PIL import Image
from skimage import color, restoration
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd as lin_svd
from scipy.signal import medfilt2d, wiener, cwt
import pywt
import cv2
import gzip

from ipfml.processing import transform, compression, segmentation
from ipfml import utils

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt


def get_image_features(data_type, block):
    """
    Method which returns the data type expected
    """

    if 'filters_statistics' in data_type:

        img_width, img_height = 200, 200

        lab_img = transform.get_LAB_L(block)
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
            # return np.mean(I_filter), np.std(I_filter)

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

    if 'statistics_extended' in data_type:

        data = get_image_features('filters_statistics', block)

        # add kolmogorov complexity
        bytes_data = np.array(block).tobytes()
        compress_data = gzip.compress(bytes_data)

        data = np.append(data, sys.getsizeof(compress_data))

        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)

        # add of svd entropy
        svd_entropy = utils.get_entropy(compression.get_SVD_s(arr))
        data = np.append(data, svd_entropy)

        # add sobel complexity (kernel size of 3)
        sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1,ksize=3)

        sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

        data = np.append(data, np.std(sobel_mag))
        
        # add sobel complexity (kernel size of 5)
        sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1,ksize=5)

        sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

        data = np.append(data, np.std(sobel_mag))

    if 'lab' in data_type:

        data = transform.get_LAB_L_SVD_s(block)

    return data


def w2d(arr, mode='haar', level=1):
    #convert to float    
    imArray = arr

    sigma = restoration.estimate_sigma(imArray, average_sigmas=True, multichannel=False)
    imArray_H = restoration.denoise_wavelet(imArray, sigma=sigma, wavelet='db1', mode='hard', 
        wavelet_levels=2, 
        multichannel=False, 
        convert2ycbcr=False, 
        method='VisuShrink', 
        rescale_sigma=True)

    # imArray_H *= 100

    return imArray_H


def _get_mscn_variance(block, sub_block_size=(50, 50)):

    blocks = segmentation.divide_in_blocks(block, sub_block_size)

    data = []

    for block in blocks:
        mscn_coefficients = transform.get_mscn_coefficients(block)
        flat_coeff = mscn_coefficients.flatten()
        data.append(np.var(flat_coeff))

    return np.sort(data)

