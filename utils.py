import os
import numpy as np
import platform

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
