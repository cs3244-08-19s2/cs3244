import os
import numpy as np
import platform
import pickle
import ioreader

def extract_image_paths(directory):
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

def create_labels(length, label):
    return [label] * length;


def load_all_imgs_within_dir(directory):
    assert os.path.exists(directory)
    
    paths = extract_image_paths(directory)

    imgs = []
    for path in paths:
        imgs.append(ioreader.load_image(path))
    imgs = np.asarray(imgs)
    
    return imgs

def cache(to_cache, cache_name):
    with open(cache_name, "wb") as f:
        pickle.dump(to_cache, f, protocol=4)
    return

def load_cache(cache_name):
    with open(cache_name, "rb") as f:
        pickle.load(f)
    return
    
def load_data(datapath, cache_name, label_name):
    imgs = []
    if (os.path.isfile(cache_name)):
        print("Using cache " + cache_name);
        imgs = load_cache(cache_name)
    else :
        print("Cache not found, loading images..")
        imgs = load_all_imgs_within_dir(datapath)
    
    labels = create_labels(len(imgs), label_name)
    return imgs, labels

                    