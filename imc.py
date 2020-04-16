import numpy as np
import cyvlfeat as vlfeat
import pickle
import math
import simplified_sift as s_sift
from scipy.spatial.distance import cdist
import cv2

import glob, os
import utils


def build_vocabulary_from_dirs(image_paths1, image_paths2, vocab_size):
    image_paths = image_paths1 +  image_paths2
    return build_vocabulary(image_paths, vocab_size)

def build_vocabulary(image_paths, vocab_size):
    """
      This function will sample SIFT descriptors from the training images,
      cluster them with kmeans, and then return the cluster centers.

      Useful functions:
      -   Use load_image_gray(path) to load grayscale images
      -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
              -  frames is a N x 2 matrix of locations, which can be thrown away
              here
              -  descriptors is a N x 128 matrix of SIFT features
            Note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster to
            compute. Also, be sure not to use the default value of step size. It
            will be very slow and you'll see relatively little performance gain
            from extremely dense sampling. You are welcome to use your own SIFT
            feature code! It will probably be slower, though.
      -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
              http://www.vlfeat.org/matlab/vl_kmeans.html
                -  X is a N x d numpy array of sampled SIFT features, where N is
                   the number of features sampled. N should be pretty large!
                -  K is the number of clusters desired (vocab_size)
                   cluster_centers is a K x d matrix of cluster centers. This is
                   your vocabulary.

      Args:
      -   image_paths: list of image paths.
      -   vocab_size: size of vocabulary

      Returns:
      -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
          cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here. You can try taking 20 features
    # per image.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    dim = 128      # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size,dim))
    total_SIFT_features = np.zeros((20*len(image_paths), dim))
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    # Obtain SIFT descriptors of each image, sample 20 from each
    n_sample = 20
    for i in range(0, len(image_paths)):
        img = load_image_gray(image_paths[i])
        frames, descriptors = vlfeat.sift.dsift(img, step=20, fast=True)
        #descriptors = s_sift.sift(img) # simplified SIFT
        desc_samples = descriptors[np.random.randint(descriptors.shape[0], size=n_sample)]
        total_SIFT_features[(i*n_sample):(i*n_sample)+(n_sample),] = desc_samples[:] 
        
    # Clustering through k_means to obtain vocabulary
    vocab = vlfeat.kmeans.kmeans(total_SIFT_features, vocab_size)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab

def bags_of_sifts(image_paths, vocab_filename):
    """
      You will want to construct SIFT features here in the same way you
      did in build_vocabulary() (except for possibly changing the sampling
      rate) and then assign each local feature to its nearest cluster center
      and build a histogram indicating how many times each cluster was used.
      Don't forget to normalize the histogram, or else a larger image with more
      SIFT features will look very different from a smaller version of the same
      image.

      Useful functions:
      -   Use load_image(path) to load RGB images and load_image_gray(path) to load
              grayscale images
      -   frames, descriptors = vlfeat.sift.dsift(img)
              http://www.vlfeat.org/matlab/vl_dsift.html
            frames is a M x 2 matrix of locations, which can be thrown away here
            descriptors is a M x 128 matrix of SIFT features
              note: there are step, bin size, and smoothing parameters you can
              manipulate for dsift(). We recommend debugging with the 'fast'
              parameter. This approximate version of SIFT is about 20 times faster
              to compute. Also, be sure not to use the default value of step size.
              It will be very slow and you'll see relatively little performance
              gain from extremely dense sampling. You are welcome to use your own
              SIFT feature code! It will probably be slower, though.
      -   D = cdist(X, Y)
            computes the distance matrix D between all pairs of rows in X and Y.
              -  X is a N x d numpy array of d-dimensional features arranged along
              N rows
              -  Y is a M x d numpy array of d-dimensional features arranged along
              N rows
              -  D is a N x M numpy array where d(i, j) is the distance between row
              i of X and row j of Y

      Args:
      -   image_paths: paths to N images
      -   vocab_filename: Path to the precomputed vocabulary.
              This function assumes that vocab_filename exists and contains an
              vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
              or visual word. This ndarray is saved to disk rather than passed in
              as a parameter to avoid recomputing the vocabulary every run.

      Returns:
      -   image_feats: N x d matrix, where d is the dimensionality of the
              feature representation. In this case, d will equal the number of
              clusters or equivalently the number of entries in each image's
              histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    for i in range(0, len(image_paths)):
        # Get SIFT descriptors
        img = load_image_gray(image_paths[i])
        frames, descriptors = vlfeat.sift.dsift(img, step=10, fast=True)
        # descriptors = s_sift.sift(img) # simplified SIFT
        # Create histogram
        dist = cdist(descriptors, vocab, 'euclidean')
        bin_assignment = np.argmin(dist, axis=1) 
        image_feats = np.zeros(200)
        for id_assign in bin_assignment:
            image_feats[id_assign] += 1
        
        feats.append(image_feats)
    
    # Normalise and convert feats into np array
    feats = np.asarray(feats)
    feats_norm_div = np.linalg.norm(feats, axis=1)
    for i in range(0, feats.shape[0]):
        feats[i] = feats[i] / feats_norm_div[i]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def load_image_gray(path):
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def load_image(path):
    return im2single(cv2.imread(path))[:, :, ::-1]

def im2single(im):
    im = im.astype(np.float32) / 255
    return im
