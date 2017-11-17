import time
import numpy as np
from scipy.misc import imresize


def preprocess_frame(frame, v_crop=(0, 0), h_crop=(0, 0)):
    """
    Preprocess image for faster computation

    Parameters
    ----------
    frame : ndarray
        Color image, of shape (H,W,C)

    v_crop : tuple, optional
        Defines how many rows of the image to remove from top and bottom, of shape (top, bottom)

    h_crop: tuple, optional
        Defines how many columns of the image to remove from left and right, of shape (left, right)

    Returns
    -------
    m : ndarray
        Greyscale image of shape(H, W)

    """

    heigth, width, _ = frame.shape
    frame = np.mean(frame, axis=2) / 255.0
    frame = frame[v_crop[0]:heigth - v_crop[1], h_crop[0]:width - h_crop[1]]
    frame = imresize(frame, size=(84, 84), interp='nearest')

    return frame


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

