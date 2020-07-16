#!/usr/bin/env python3

"""
Wake Word TensorFlow Lite Conversion

Script that converts a Keras model file (.h5) into a TensorFlow Lite file
(.tflite) for use with TensorFlow Lite.

You will need the following packages (install via pip):
 * numpy
 * tensorflow

Example call:
python 05_tflite_conversion.py -i "./model.h5" -o "./model.tflite"

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

import argparse
from os import remove
from os.path import join, exists

import numpy as np
import tensorflow as tf
from tensorflow import lite
from tensorflow.keras import models

import utils

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2020, Shawn Hymel"
__license__ = "MIT"
__version__ = "0.1"

################################################################################
# Main

###
# Parse command line arguments

# Script arguments
parser = argparse.ArgumentParser(description="Converts .h5 Keras model file to "
                                    "a .tflite TensorFlow Lite file.")
parser.add_argument('-i', 
                    '--in_file', 
                    action='store', 
                    dest='in_file',
                    type=str,
                    required=True,
                    help="Path to Keras (.h5) file")
parser.add_argument('-o',
                    '--out_file',
                    action='store',
                    type=str,
                    default="./model.tflite",
                    help="Path to store TensorFlow Lite file (default: "
                            "./model.tflite)")

# # Parse arguments
args = parser.parse_args()
in_file = args.in_file
out_file = args.out_file

###
# Welcome screen

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

# Convert modele to TensorFlow Lite model
model = models.load_model(in_file)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
new_filename = out_file
while exists(new_filename):
    print("WARNING: TFLite Model file already exists")
    print(new_filename)
    resp = utils.query_yes_no("Overwrite file?")
    if resp:
        print("Overwriting file")
        remove(new_filename)
        break
    else:
        new_filename += "_.tflite"
        print("Trying:", new_filename)
open(new_filename, 'wb').write(tflite_model)
print("TFLite model saved:", new_filename)

# Say we're done
print("Done!")
exit()