import cv2
import numpy as np
from skimage import io
import os.path as osp

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)

def im2single(im):
    im = im.astype(np.float32) / 255
    return im

def load_image(path):
    return cv2.imread(path)

def load_image_gray(path):
    img = im2single(cv2.imread(path))[:, :, ::-1]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)