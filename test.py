import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


from helpers import sigmoid

# Load data
# You are given a dataset ("data.h5") containing:
# - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
# - a test set of m_test images labeled as cat or non-cat
# - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
# We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).
# Each line of your train_set_x_orig and test_set_x_orig is an array representing an image.
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Reshape data
# For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px  ∗∗  num_px  ∗∗  3, 1). After this, our training (and test) dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize data
# One common preprocessing step in machine learning is to center and standardize your dataset,
# meaning that you substract the mean of the whole numpy array from each example, and then
# divide each example by the standard deviation of the whole numpy array. But for picture
# datasets, it is simpler and more convenient and works almost as well to just divide every
# row of the dataset by 255 (the maximum value of a pixel channel).
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

