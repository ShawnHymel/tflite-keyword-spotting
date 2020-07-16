#!/usr/bin/env python3

"""
Wake Word Training

Script that uses TensorFlow to train a 2D convolutional neural network to
classify STFTs computed from spoken words (or background noise).

You will need the following packages (install via pip):
 * numpy
 * tensorflow

Example call:

The MIT License (MIT)

Copyright (c) 2020 Shawn Hymel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import argparse
from os import makedirs, listdir, rename
from os.path import isdir, join, exists

import numpy as np
import tensorflow as tf

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2020, Shawn Hymel"
__license__ = "MIT"
__version__ = "0.1"

# Settings
acceptable_feature_versions = [0.1]

################################################################################
# Functions

################################################################################
# Main

###
# Parse command line arguments

# Script arguments
parser = argparse.ArgumentParser(description="Neural network training tool "
                                "reads in a collection of STFT arrays and runs "
                                "training algorithm(s). Will train multiple "
                                "NNs and choose best one (based on F1 score).")
parser.add_argument('-d', 
                    '--in_dir', 
                    action='store', 
                    dest='in_dir',
                    type=str,
                    required=True,
                    help="")

# Parse arguments
args = parser.parse_args()
in_dir = args.in_dir

###
# Welcome screen

# Print tool welcome message
print("-----------------------------------------------------------------------")
print("Wake Word Training Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")
print("Using GPU support: " + str(tf.test.is_built_with_cuda()))

# Limit GPU memory growth:
# https://github.com/tensorflow/tensorflow/issues/25160
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=
                1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

###
# Perform training

# Read in feature sets
labels = []
x_all = []
y_all = []
for i, filename in enumerate(listdir(in_dir)):
    npz_data = np.load(join(in_dir, filename))

    # Check to make sure we're working with an acceptable version of features
    if not float(npz_data['version']) in acceptable_feature_versions:
        print(  "ERROR: Feature version mismatch. Feature set " + 
                str(filename) + " is version " + str(npz_data['version']) + 
                ". Feature sets need to be one of the following versions: " + 
                str(acceptable_feature_versions))
        exit()

    # Add to filename to list of labels
    labels.append(filename.replace(".npz", ""))

    # Add samples to our x set
    x_all.append(npz_data['samples'])

    # Get number of samples in each file and fill y label list
    num_samples = npz_data['samples'].shape[0]
    y_all.append([i] * num_samples)

# Show the labels that we found
print("Labels: " + str(labels))

# Convert X and y sets into a Numpy arrays
x_all = np.concatenate(x_all)
y_all = np.concatenate(y_all)

#TODO: Zip and shuffle. Break into training, validation, test sets.
