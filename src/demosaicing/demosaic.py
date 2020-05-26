import numpy as np

from scipy.ndimage import convolve2d

from .constants import filter0
from .constants import filter1
from .constants import filter2
from .constants import filter3


def demosaic(raw_bayer):
    # height, width = raw_bayer.shape

    # d0 = correlate(raw_bayer, filter0, mode='constant', cval=0.0) / np.sum(filter0)
    # d1 = correlate(raw_bayer, filter1, mode='constant', cval=0.0) / np.sum(filter1)
    # d2 = correlate(raw_bayer, filter2, mode='constant', cval=0.0) / np.sum(filter2)
    # d3 = correlate(raw_bayer, filter3, mode='constant', cval=0.0) / np.sum(filter3)

    # out = np.zeros(height // 2, width // 2, 3)

    pass




