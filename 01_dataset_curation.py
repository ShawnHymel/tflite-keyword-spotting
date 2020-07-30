#!/usr/bin/env python3

"""
Dataset Curation

Script to select which audio files should be included in the training/test 
dataset. This allows you to limit the total number of files (faster training but
lower accuracy).

You will need the following packages (install via pip):
 * shutil

Example call:
python .\01_dataset_curation.py -m 100 
    -o "../../Python/datasets/custom_wake_words_curated"
    "../../Python/datasets/custom_wake_words_edited" 
    "../../Python/datasets/speech_commands_dataset"

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
import random
import argparse
from os import makedirs, listdir, rename
from os.path import isdir, join, exists

import shutil

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
parser = argparse.ArgumentParser(description="Audio file curation tool that "
                                "copies files from multiple source directories "
                                "to an output directory. The script will also "
                                "format the subdirectories to make the next "
                                "script in the chain run more easily.")
parser.add_argument('-m', 
                    '--max', 
                    action='store', 
                    dest='max',
                    type=int,
                    default=0,
                    help="Maximum number of files (randomly chosen) to be "
                            "included in curated dataset. 0 means keep all "
                            "files found in each director (default: 0)")
parser.add_argument('-o',
                   '--out_dir',
                   action='store',
                   dest='out_dir',
                   type=str,
                   required=True,
                   help="Directory where the curated audio samples are to be "
                        "stored")
parser.add_argument('directories',
                    metavar='d',
                    type=str,
                    nargs='+',
                    help="List of source directories to include")

# Parse arguments
args = parser.parse_args()
max_files = args.max
out_dir = args.out_dir
in_dirs = args.directories

###
# Welcome screen

# Print tool welcome message
print("-----------------------------------------------------------------------")
print("Dataset Curation Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")

###
# Set up directories

# Delete output directory if it already exists
if isdir(out_dir):
    print("WARNING: Output directory already exists:")
    print(out_dir)
    print("This tool will delete the output directory and everything in it.")
    resp = utils.query_yes_no("Continue?")
    if resp:
        print("Deleting and recreating output directory.")
        #rename(out_dir, out_dir + '_') # One way to deal with OS blocking rm
        shutil.rmtree(out_dir)
        time.sleep(2.0)
    else:
        print("Please delete directory to continue. Exiting.")
        exit()

# Create output dir
if not exists(out_dir):
    makedirs(out_dir)
else:
    print("ERROR: Output directory could not be deleted. Exiting.")
    exit()

# Create a list of possible words from the subdirectories
word_list = []
for directory in in_dirs:
    for name in listdir(directory):
        if isdir(join(directory, name)):
            word_list.append(name)

# Remove duplicates from list
word_list = list(dict.fromkeys(word_list))

# Go through each word and randomly choose files for that word
for word in word_list:

    # Construct a full list of available files (full paths)
    word_paths = []
    for directory in in_dirs:
        sub_dir = join(directory, word)
        if isdir(sub_dir):
            for filename in listdir(sub_dir):
                word_paths.append(join(sub_dir, filename))

    # Shuffle list and choose first m files (include all if m is 0)
    random.shuffle(word_paths)
    if max_files > 0:
        word_paths = word_paths[:max_files]

    # If there are no files, skip
    num_files = len(word_paths)
    if num_files == 0:
        print("No files found for " + word + ". Skipping.")
        continue


    # Create subdirectory in output directory
    out_subdir = join(out_dir, word)
    makedirs(out_subdir)

    # Show progress bar
    print("Copying samples for " + word)
    utils.print_progress_bar(   0, 
                                num_files, 
                                prefix="Progress:", 
                                suffix="Complete", 
                                length=50)

    # Copy m files to subdirectory, renaming to avoid conflict
    digit_cnt = len(str(len(word_paths)))
    for i, filepath in enumerate(word_paths):

        # Construct filename (prepend with 0s) and copy
        filename = str(i).zfill(digit_cnt) + ".wav"
        shutil.copy(filepath, join(out_subdir, filename))

        # Update progress bar
        utils.print_progress_bar(   i + 1, 
                                    num_files, 
                                    prefix="Progress:", 
                                    suffix="Complete", 
                                    length=50)

# Say we're done
print("Done!")
exit()