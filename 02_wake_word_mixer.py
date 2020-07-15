#!/usr/bin/env python3

"""
Wake Word Mixer

Script that reads in various spoken word samples, resamples them, and
combines them with random snippets of background noise. Samples that are too
long are truncated at the end, and samples that are too short are padded with
0s (also at the end). Augmented samples are stored in the specified directory.

Note that all audio is mixed to mono.

Output files are not randomized/shuffled, which makes it easier to check the
output versus the input sounds. They are simply numbered in the order in which
they are read

You will need the following packages (install via pip):
 * numpy
 * librosa
 * soundfile
 * shutil

Example call:
python 02_wake_word_mixer.py -d "../../Python/datasets/custom_wake_words_edited"
    -b "../../Python/datasets/ambient/sounds" 
    -o "../../Python/datasets/custom_wake_words_mixed" 
    -t "how_are_you, goodnight" -w 1.0 -g 0.5 -s 1.0 -r 16000 -e PCM_16
    -n 5

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

import random
import argparse
from os import makedirs, listdir, rename
from os.path import isdir, join, exists

import shutil
import librosa
import numpy as np
import soundfile as sf

import utils

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2020, Shawn Hymel"
__license__ = "MIT"
__version__ = "0.1"

# Settings
other_dir_name = "_other"
bg_dir_name = "_background"

################################################################################
# Functions

# Mix audio and random snippet of background noise
def mix_audio(word_path=None, 
                bg_path=None, 
                word_vol=1.0, 
                bg_vol=1.0, 
                sample_time=1.0,
                sample_rate=16000):
    """
    Read in a wav file and background noise file. Resample and adjust volume as
    necessary.
    """
    
    # If no word file is given, just return random background noise
    if word_path == None:
        waveform = [0] * int(sample_time * sample_rate)
        fs = sample_rate
    else:

        # Open wav file, resample, mix to mono
        waveform, fs = librosa.load(word_path, sr=sample_rate, mono=True)
        
        # Pad 0s on the end if not long enough
        if len(waveform) < sample_time * sample_rate:
            waveform = np.append(waveform, np.zeros(int((sample_time * 
                sample_rate) - len(waveform))))

        # Truncate if too long
        waveform = waveform[:int(sample_time * sample_rate)]

    # If no background noise is given, just return the waveform
    if bg_path == None:
        return waveform

    # Open background noise file
    bg_waveform, fs = librosa.load(bg_path, sr=fs)

    # Pick a random starting point in background file
    max_end = len(bg_waveform) - int(sample_time * sample_rate)
    start_point = random.randint(0, max_end)
    end_point = start_point + int(sample_time * sample_rate)
    
    # Mix the two sound samples (and multiply by volume)
    waveform = [0.5 * word_vol * i for i in waveform] + \
                (0.5 * bg_vol * bg_waveform[start_point:end_point])

    return waveform

# Go through each wave file in given directory, mix with bg, save to new file
def mix_files(word_dir, bg_dir, out_dir, num_file_digits=None, start_cnt=0):
    """
    Go through each wav file in word_dir, mix each with each wav file in bg_dir
    and save to new file in out_dir.
    
    Returns the number of files created.
    """

    # Figure out number of digits so the filenames line up in OS
    num_word_files = len(listdir(word_dir))
    num_bg_files = len(listdir(bg_dir))
    num_output_files = num_word_files * num_bg_files
    digit_cnt = len(str(num_output_files))

    # We can't handle no output files
    if num_output_files == 0:
        return 0

    # If we're not given a file digits length, use the one we just calculated
    if num_file_digits is None:
        num_file_digits = digit_cnt

    # Show progress bar
    print(str(num_word_files) + " word files * " + str(num_bg_files) + 
            " bg files = " + str(num_output_files) + " output files")
    utils.print_progress_bar(   0, 
                                num_output_files, 
                                prefix="Progress:", 
                                suffix="Complete", 
                                length=50)

    # Go through each target word, mixing it with background noise
    file_cnt = 0
    for word_filename in listdir(word_path):
        for bg_filename in listdir(bg_dir):

            # Mix word with background noise
            waveform = mix_audio(word_path=join(word_path, word_filename), 
                                bg_path=join(bg_dir, bg_filename), 
                                word_vol=word_vol, 
                                bg_vol=bg_vol, 
                                sample_time=sample_time,
                                sample_rate=sample_rate)

            # Save to new file
            filename = str(start_cnt + file_cnt).zfill(num_file_digits) + '.wav'
            sf.write(join(out_subdir, filename), 
                    waveform, 
                    sample_rate, 
                    subtype=bit_depth)
            file_cnt += 1

            # Update progress bar
            utils.print_progress_bar(   file_cnt + 1, 
                                        num_output_files, 
                                        prefix="Progress:", 
                                        suffix="Complete", 
                                        length=50)
    
    # Return file count
    print()
    return file_cnt

################################################################################
# Main

###
# Parse command line arguments

# Script arguments
parser = argparse.ArgumentParser(description="Audio mixing tool that "
                                "combines spoken word examples with random "
                                "bits of background noise to create a dataset "
                                "for use with wake word training.")
parser.add_argument('-d', 
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
parser.add_argument('-w',
                    '--word_vol',
                    action='store',
                    dest='word_vol',
                    type=float,
                    default=1.0,
                    help="Relative volume to multiply each word by (default: "
                         "1.0)")
parser.add_argument('-g',
                    '--bg_vol',
                    action='store',
                    dest='bg_vol',
                    type=float,
                    default=1.0,
                    help="Relative volume to multiply each background noise by "
                            "(default: 1.0)")
parser.add_argument('-s',
                    '--sample_time',
                    action='store',
                    dest='sample_time',
                    type=float,
                    default=1.0,
                    help="Time (seconds) of each output clip (default: 1.0)")
parser.add_argument('-r',
                    '--sample_rate',
                    action='store',
                    dest='sample_rate',
                    type=int,
                    default=16000,
                    help="Sample rate (Hz) of each output clip (default: "
                            "16000)")
parser.add_argument('-e',
                    '--bit_depth',
                    action='store',
                    dest='bit_depth',
                    type=str,
                    choices=['PCM_16', 'PCM_24', 'PCM_32', 'PCM_U8', 'FLOAT',
                             'DOUBLE'],
                    default='PCM_16',
                    help="Bit depth of each sample (default: PCM_16)")
parser.add_argument('-n',
                    '--num_bg_samples',
                    action='store',
                    dest='num_bg_samples',
                    type=int,
                    default=3,
                    help="Number of random clips to take from each background "
                            "noise file (default: 3)")

# Parse arguments
args = parser.parse_args()
words_dir = args.words_dir
bg_dir = args.bg_dir
out_dir = args.out_dir
targets = args.targets
word_vol = args.word_vol
bg_vol = args.bg_vol
sample_time = args.sample_time
sample_rate = args.sample_rate
bit_depth = args.bit_depth
num_bg_samples = args.num_bg_samples

###
# Welcome screen

# Print tool welcome message
print("-----------------------------------------------------------------------")
print("Wake Word Mixer Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")

###
# Set up directories

# Create a list of possible words from the subdirectories
word_list = [name for name in listdir(words_dir)]
print("Number of words found:", len(word_list))

# Make target list and make sure each target word appears in our list of words
target_list = [word.strip() for word in targets.split(',')]
for target in target_list:
    if target not in word_list:
        print("ERROR: Target word '" + target + "' not found as subdirectory "
                "in words directory. Exiting.")
        exit()

# Remove duplicate targets
target_list = list(dict.fromkeys(target_list))

# Remove targets from word list to create "other" list
other_list = []
[other_list.append(name) for name in word_list if name not in target_list]

# Delete output directory if it already exists
if isdir(out_dir):
    print("WARNING: Output directory already exists:")
    print(out_dir)
    print("This tool will delete the output directory and everything in it.")
    resp = utils.query_yes_no("Continue?")
    if resp:
        print("Deleting and recreating output directory.")
        rename(out_dir, out_dir + '_')
        shutil.rmtree(out_dir + '_')
    else:
        print("Please delete directory to continue. Exiting.")
        exit()

# Create output dir
if not exists(out_dir):
    makedirs(out_dir)
else:
    print("ERROR: Output directory could not be deleted. Exiting.")
    exit()

###
# Save clips of background noise

# Create _background subdirectory
out_subdir = join(out_dir, bg_dir_name)
makedirs(out_subdir)

# Go through each file, grabbing some snippets from each
num_bg_files = len(listdir(bg_dir))
num_bg_clips = num_bg_samples * num_bg_files
digit_cnt = len(str(num_bg_clips))

# Print what we're doing and show progress bar
print("Gathering random background noise snippets")
print(str(num_bg_samples) + " samples per file * " + str(num_bg_files) + 
        " bg files = " + str(num_bg_clips) + " output files")
utils.print_progress_bar(   0, 
                            num_bg_clips, 
                            prefix="Progress:", 
                            suffix="Complete", 
                            length=50)

# Go through each background file, taking a random sample from different points
file_cnt = 0
for bg_filename in listdir(bg_dir):
    for i in range(num_bg_samples):

        # Get random snippet from background noise
        waveform = mix_audio(word_path=None, 
                                bg_path=join(bg_dir, bg_filename), 
                                word_vol=word_vol, 
                                bg_vol=bg_vol, 
                                sample_time=sample_time,
                                sample_rate=sample_rate)
        
        # Save to new file
        filename = str(file_cnt).zfill(digit_cnt) + '.wav'
        sf.write(join(out_subdir, filename), 
                waveform, 
                sample_rate, 
                subtype=bit_depth)
        file_cnt += 1

        # Update progress bar
        utils.print_progress_bar(   file_cnt + 1, 
                                    num_bg_clips, 
                                    prefix="Progress:", 
                                    suffix="Complete", 
                                    length=50)

# Newline to exit progress bar
print()

###
# Mix target words

# Start with target words
for target in target_list:

    # Create output subdirectory with the target word's name
    out_subdir = join(out_dir, target)
    makedirs(out_subdir)

    # Create full path to word directory
    word_path = join(words_dir, target)

    # Start the mixing machine
    print("Mixing: '" + str(target) +"'")
    mix_files(word_path, bg_dir, out_subdir)

###
# Mix words in "other" category

# Figure out total number of words in other category
num_word_files = 0
for word in other_list:
    word_path = join(words_dir, word)
    num_word_files += len(listdir(word_path))

# Calculate number of digits for our output files in other
num_bg_files = len(listdir(bg_dir))
num_output_files = num_word_files * num_bg_files
digit_cnt = 1
while(num_output_files > 1):
    num_output_files = num_output_files // 10
    digit_cnt += 1

# Mix together other files into "other" subdirectory
out_subdir = join(out_dir, other_dir_name)
makedirs(out_subdir)
last_file_num = 0
for word in other_list:
    word_path = join(words_dir, word)
    print("Mixing: '" + str(word) + "' to " + other_dir_name)
    last_file_num = last_file_num + mix_files(word_path, 
                                                bg_dir, 
                                                out_subdir, 
                                                digit_cnt, 
                                                last_file_num)

# Say we're done
print("Done!")
exit()