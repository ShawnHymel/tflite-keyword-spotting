#!/usr/bin/env python3

"""
Wake Word Training

Script that uses TensorFlow to train a 2D convolutional neural network to
classify STFTs computed from spoken words (or background noise).

You will need the following packages (install via pip):
 * numpy
 * tensorflow

Example call:
python 04_training.py -d "../../Python/datasets/custom_wake_words_features" 
    -o "./model.h5" -v 0.2 -t 0.2 -n 20

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

import math
import time
import random
import argparse
from os import makedirs, listdir, rename, remove
from os.path import isdir, join, exists

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, backend

import utils

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2020, Shawn Hymel"
__license__ = "MIT"
__version__ = "0.1"

# Settings
bg_label = "_background"
other_label = "_other"
acceptable_feature_versions = [0.1]
optimizer = 'adam'      # Optimizer algorithm to use
loss_function = 'sparse_categorical_crossentropy'
num_epochs = 200
batch_size = 100

################################################################################
# Functions

# Compute confusion matrix manually. From:
# https://stackoverflow.com/a/48087308/12092748
def compute_confusion_matrix(true, pred):
  """
  Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.
  """

  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result

# Print a pretty version of the confusion matrix
def print_confusion_matrix(cm, labels):
    """
    Prints a much nicer version of the confusion matrix,
    complete with labels.
    """
    
    # Figure out how much space label column needs
    p_header = "Prediction: "
    label_col_w = max([max([len(s) for s in labels]), len(p_header)])

    # Figure out how much space label index needs
    idx_col_w = len(str(len(labels)))

    # Figure out how much space each column needs
    max_val = int(np.max(cm))
    num_digits = len(str(max_val))
    col_w = max([num_digits + 2, 5])

    # Print out a pretty version of the confusion matrix
    print("Actual:")
    for i, cm_row in enumerate(cm):
        print("{:>{col_w}} ({:^{idx_w}}) ".format(labels[i], i, 
                idx_w=idx_col_w, col_w=label_col_w), end='')
        for val in cm_row:
            print("{:^{col_w}}".format(int(val), col_w=col_w), end='')
        print()

    # Print out bottom row headers
    space_w = label_col_w + idx_col_w + 4
    print("{:>{col_w}}".format(p_header, col_w=space_w), end='')
    for i in range(len(labels)):
        print("{:^{col_w}}".format("(" + str(i) + ")", col_w=col_w), end='')
    
    # Make sure we add a newline
    print()

# Calculate false positive and false negative rates from a confusion matrix
def get_fpr_fnr(cm, idx):
    fpr = np.sum(np.delete(cm[:, idx], idx)) / np.sum(cm[:, idx])
    fnr = np.sum(np.delete(cm[idx, :], idx)) / np.sum(cm[idx, :])
    return fpr, fnr

# Calculate F1 score of a particular index
def get_f1_score(cm, idx):
    precision = cm[(idx, idx)] / np.sum(cm[:, idx])
    recall = cm[(idx, idx)] / np.sum(cm[idx, :])
    f1_score = 2 * (precision * recall) / (precision + recall)
    if math.isnan(f1_score):
        return 0.0
    return f1_score

# Function to build our model
def build_model():
    
    # OG model we started with
    model = models.Sequential([
        
        # Reshape input
        layers.InputLayer(input_shape=sample_shape),
        layers.Reshape(target_shape=(sample_shape + (1,))),
        
        # Convolutional layer
        layers.Conv2D(filters=30, kernel_size=(5, 5), activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.MaxPooling2D(pool_size=(12, 12), padding='same'),
        
        # Flatten
        layers.Flatten(),
        
        # Classifier
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(labels), activation=tf.nn.softmax)
    ])
    
    return model

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
                    help="Directory where the .npz feature sets are stored.")
parser.add_argument('-o',
                    '--out_file',
                    action='store',
                    type=str,
                    default="./model.h5",
                    help="Path to store trained model as a Keras (.h5) file "
                            "(default: ./model.h5)")
parser.add_argument('-v',
                    '--val_ratio',
                    action='store',
                    dest='val_ratio',
                    type=float,
                    default=0.2,
                    help="Ratio of validation samples to use out of whole set "
                            "(default: 0.2)")
parser.add_argument('-t',
                    '--test_ratio',
                    action='store',
                    dest='test_ratio',
                    type=float,
                    default=0.2,
                    help="Ratio of test samples to use out of whole set "
                            "(default: 0.2)")
parser.add_argument('-n',
                    '--num_train',
                    action='store',
                    dest='num_train',
                    type=int,
                    default=20,
                    help="Number of random model initializations to try to "
                            "find model with best average F1 score across "
                            "target words (default: 20)")

# # Parse arguments
args = parser.parse_args()
in_dir = args.in_dir
out_file = args.out_file
val_ratio = args.val_ratio
test_ratio = args.test_ratio
num_train = args.num_train

###
# Welcome screen

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

# Print tool welcome message
print("-----------------------------------------------------------------------")
print("Wake Word Training Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")
print("Using GPU support: " + str(tf.test.is_built_with_cuda()))

# Print versions
print('Numpy ' + np.__version__)
print('TensorFlow ' + tf.__version__)
print('Keras ' + tf.keras.__version__)

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

# Print out accuracy if we just guess everything is '_other'
other_idx = labels.index(other_label)
num_other = np.sum(1 * (y_all == other_idx))
print("Accuracy if we guess everything is '_other': " + 
        str(round(100 * num_other / y_all.shape[0], 2)) + "%") 
print("    (we need to do better than this)")

# Zip, shuffle, unzip
zipped_set = list(zip(x_all, y_all))
random.shuffle(zipped_set)
x_all, y_all = zip(*zipped_set)
x_all = np.asarray(x_all)
y_all = np.asarray(y_all)

# Split feature sets into training, validation, and test sets
val_set_size = int(x_all.shape[0] * val_ratio)
test_set_size = int(x_all.shape[0] * test_ratio)
x_val = x_all[:val_set_size]
y_val = y_all[:val_set_size]
x_test = x_all[val_set_size:(val_set_size + test_set_size)]
y_test = y_all[val_set_size:(val_set_size + test_set_size)]
x_train = x_all[(val_set_size + test_set_size):]
y_train = y_all[(val_set_size + test_set_size):]

# Input shape for CNN is size of STFT of 1 sample
sample_shape = x_train.shape[1:]

# Get a list of our target classes (as numbers)
other_idx = labels.index(other_label)
bg_idx = labels.index(bg_label)
target_idxs = list(range(len(labels)))
target_idxs.remove(other_idx)
target_idxs.remove(bg_idx)
print('Target words:', [labels[idx] for idx in target_idxs])

# Train several models and choose model with highest F1 score
best_model = None
best_f1_score = 0.0

# Try taining a bunch of models and keeping the best one
for i in range(num_train):
    
    # Keep track of which loop we're on
    print("Evaluating model " + str(i + 1) + "/" + str(num_train) + "...")
    
    # Create model (need to do this every time to re-initialize random weights)
    model = build_model()
    
    # Ze train!
    model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=['acc'])
    history = model.fit(x_train,
                        y_train, 
                        epochs=num_epochs, 
                        batch_size=batch_size,
                        verbose=0,
                        validation_data=(x_val, y_val))
    
    # Evaluate model on validation set
    predictions = model.predict(x_val)
    y_hat = [np.argmax(prediction) for prediction in predictions]
    correct_list = (np.array(y_hat) == y_val) * 1
    
    # Make confusion matrix
    cm = compute_confusion_matrix(y_val, y_hat)
    
    # Compute average FPR, FNR, F1 over target words we care about
    fpr_avg = 0.0
    fnr_avg = 0.0
    f1_avg = 0.0
    for idx in target_idxs:
        fpr, fnr = get_fpr_fnr(cm, idx)
        f1 = get_f1_score(cm, idx)
        fpr_avg += fpr / len(target_idxs)
        fnr_avg += fnr / len(target_idxs)
        f1_avg += f1 / len(target_idxs)
    
    # Print out metrics
    print(cm)
    print("Average false positive rate:", fpr_avg)
    print("Average false negative rate:", fnr_avg)
    print("Average F1 score:", f1_avg)
    print("-------------------------------------------------------------------")
    
    # Compare to best score and save model of best score
    if f1_avg > best_f1_score:
        best_f1_score = f1_avg
        best_model = model
        
print()
print("Model evaluation done. Best score:", best_f1_score)
print()

# Combine training and validation sets into one large set
x_train_val = np.concatenate((x_train, x_val))
y_train_val = np.concatenate((y_train, y_val))

# Train chosen model with larger set
print("Training chosen model with full (training and validation) sets...")
model.compile(optimizer=optimizer,
          loss=loss_function,
          metrics=['acc'])
history = model.fit(x_train_val,
                    y_train_val, 
                    epochs=num_epochs, 
                    batch_size=batch_size,
                    verbose=0)
print("Model training done")

# Evaluate model on test set
predictions = model.predict(x_test)
y_hat = [np.argmax(prediction) for prediction in predictions]
correct_list = (np.array(y_hat) == y_test) * 1

# Make confusion matrix
cm = compute_confusion_matrix(y_test, y_hat)

# Compute average FPR, FNR, F1 over target words we care about
fpr_avg = 0.0
fnr_avg = 0.0
f1_avg = 0.0
for idx in target_idxs:
    fpr, fnr = get_fpr_fnr(cm, idx)
    f1 = get_f1_score(cm, idx)
    fpr_avg += fpr / len(target_idxs)
    fnr_avg += fnr / len(target_idxs)
    f1_avg += f1 / len(target_idxs)

# Print out results from test set
print()
print("Evaluation results from test set")
print("Confusion matrix")
print("---")
print_confusion_matrix(cm, labels)
print("---")
print("Average false positive rate of target words:", fpr_avg)
print("Average false negative rate of target words:", fnr_avg)
print("Average F1 score of target words:", f1_avg)

# Save model
new_filename = out_file
while exists(new_filename):
    print("WARNING: Model file already exists")
    print(new_filename)
    resp = utils.query_yes_no("Overwrite file?")
    if resp:
        print("Overwriting file")
        remove(new_filename)
        break
    else:
        new_filename += "_.h5"
        print("Trying:", new_filename)
model.save(new_filename)
print("Model saved:", new_filename)

# Say we're done
print("Done!")
exit()