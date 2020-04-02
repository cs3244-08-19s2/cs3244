import os
import cv2
import numpy as np
import platform
import pickle
import ioreader
from keras.utils import normalize

def extract_image_paths(directory: str):
    """
    Extract all the absolute paths to *.jpg inside the directory.

    Parameters:
        directory (str): directory that stores the *.jpg images

    Returns:
        arr of str: array of absolute paths to *.jpg

    """
    assert os.path.exists(directory)

    dir_separator = "/" if platform.system() == "Windows" else "\\"
    # Find individual image paths
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            image_paths.append(os.path.abspath(directory) + dir_separator + filename)

    return image_paths


def create_labels(length: int, label: str):
    """
    Create a list of labels with length of list equal to the argument given.
    
    Parameters:
        length (int): total number of labels to be created
        label (str): label name
    
    Returns:
        arr of str: array of labels
    """
    assert length >= 0
    
    return [label] * length;


def load_all_imgs_within_dir(directory: str):
    """
    Load all the *.jpg contained within the directory

    Parameters:
        directory (str): directory that stores the *.jpg images

    Returns:
        numpy.array of img: array of images
    """
    paths = extract_image_paths(directory)

    imgs = []
    for path in paths:
        imgs.append(ioreader.load_image(path))
    imgs = np.asarray(imgs)
    
    return imgs


def cache(to_cache, cache_name: str):
    """
    Cache the numpy.array input into a cache file
    
    Parameters:
        to_cache (numpy.arr): object to cache
        cache_name (str): name of cache file
    """
    with open(cache_name, "wb") as f:
        pickle.dump(to_cache, f, protocol=4)
    return


def load_cache(cache_name):
    """
    Load the cache file 
    
    Parameters:
        cache_name (str): name of cache file
    """
    return pickle.load(open(cache_name, "rb"))
    
    
def load_data(datapath:str, cache_name: str, label_name: str):
    """
    Load all the *.jpg from the given data path and create 
    the appropriate labels for the data as well.
    
    Alternatively, this method can also load the *.jpg file from a cache 
    file, identified by the cache_name.
    
    Parameters:
        datapath (str): directory that stores the *.jpg images
        cache_name (str): name of cache file
        label_name (str): label name
    
    Returns:
        imgs (numpy.array): array of images
        labels (arr of str): array of labels
    """
    imgs = []
    if (os.path.isfile(cache_name)):
        print("Using cache " + cache_name);
        imgs = load_cache(cache_name)
    else :
        print("Cache not found, loading images..")
        imgs = load_all_imgs_within_dir(datapath)
    
    labels = create_labels(len(imgs), label_name)
    return imgs, labels


def images_crop_from_centre(images, percentage=0.9):
    cropped_images = []
    for i in range(len(images)):
        H, W, C = images[i].shape
        delta_H_new, delta_W_new = int(H * percentage * 0.5), int(W * percentage * 0.5)
        centre_H, centre_W = int(H/2), int(W/2)
        crop_by = delta_H_new if delta_H_new < delta_W_new else delta_W_new
        cropped_images.append(images[i][centre_H - crop_by : centre_H + crop_by, 
                                        centre_W - crop_by : centre_W + crop_by, :])
    return np.asarray(cropped_images)


def images_resize(images, dim: (int, int)):
    resized_images = []
    for i in range(len(images)):
         resized_images.append(cv2.resize(images[i], dim))
    return np.asarray(resized_images).reshape(len(images), dim[0], dim[1], images[0].shape[2])


def images_normalize(images):
    normalised_images = []
    for i in range(len(images)):
        normalised_images.append(cv2.normalize(images[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    return np.asarray(normalised_images).reshape(len(images), images[0].shape[0], images[0].shape[1], images[0].shape[2])