import cv2, pywt
import numpy as np
from scipy.signal import medfilt2d, wiener, cwt

def w2d(arr, mode='haar', level=1):
    #convert to float   
    imArray = arr
    imArray /= 255

    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H