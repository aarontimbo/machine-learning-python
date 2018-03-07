# coursera notebook - Week 4
# Convolutional Neural Network
# Art Generation with Neural Style Transfer

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

%matplotlib inline

# Load pretrained VGG-19 (19 layer) network
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)

# GRADED FUNCTION: compute_content_cost

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(tf.transpose(a_C, perm=[0, 3, 1, 2]), [m, n_C, n_H * n_C])
    print('a_C_unrolled shape: ' + str(a_C_unrolled.shape))
    a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), [m, n_C, n_H * n_C])
    print('a_G_unrolled shape: ' + str(a_G_unrolled.shape))
    
    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(a_C_unrolled) - tf.reduce_sum(a_G_unrolled)
    ### END CODE HERE ###
    
    return J_content

