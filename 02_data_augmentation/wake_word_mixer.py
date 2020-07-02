# Wake Word Mixer
#
# Author: Shawn Hymel
# Date: July 2, 2020
#
# Script that reads in various spoken word samples, resamples them, and
# combines them with random snippets of background noise. Samples that are too
# long are truncated at the end, and samples that are too short are padded with
# 0s (also at the end). Augmented samples are stored in the specified directory.
#
# The MIT License (MIT)
#
# Copyright (c) 2020 Shawn Hymel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from os import makedirs, listdir
from os.path import isdir, join, exists

# Specify version of this tool
__version__ = "0.1"

################################################################################
# Functions

################################################################################
# Main

# Script arguments
parser = argparse.ArgumentParser(description="Audio mixing tool that "
                                "combines spoken word examples with random "
                                "bits of background noise to create a dataset "
                                "for use with wake word training.")
parser.add_argument('-w', 
                    '--words_dir', 
                    action='store', 
                    dest='words_dir',
                    type=str,
                    required=True,
                    help="Directory where the spoken word samples are stored. "
                        "Subdirectories in this directory should be named "
                        "after the word being spoken in the samples. "
                        "Subdirectories with the same names as those given by "
                        "the 'targets' option must be present.")
parser.add_argument('-b',
                   '--bg_dir',
                   action='store',
                   dest='bg_dir',
                   type=str,
                   required=True,
                   help="Directory where the background noise clips are stored")
parser.add_argument('-o',
                   '--out_dir',
                   action='store',
                   dest='out_dir',
                   type=str,
                   required=True,
                   help="Directory where the mixed audio samples are to be "
                        "stored")
parser.add_argument('-t',
                   '--targets',
                   action='store',
                   dest='targets',
                   type=str,
                   required=True,
                   help="List of target words, separated by commas. Spaces not "
                        "allowed--they will be treated as commas.")

# Parse arguments
args = parser.parse_args()
words_dir = args.words_dir
bg_dir = args.bg_dir
targets = args.targets

# Print tool message
print("-----------------------------------------------------------------------")
print("Wake Word Mixer Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")

# Create a list of possible words from the subdirectories
word_list = [name for name in listdir(words_dir)]

# Make target list and make sure each target word appears in our list of words
target_list = [word.strip() for word in targets.split(',')]
for target in target_list:
    if target not in word_list:
        print("ERROR: Target word '" + target + "' not found as subdirectory "
                "in words directory. Exiting.")
        exit()
