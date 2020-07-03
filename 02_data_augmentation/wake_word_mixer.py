# Wake Word Mixer
#
# Author: Shawn Hymel
# Date: July 2, 2020
#
# Script that reads in various spoken word samples, resamples them, and
# combines them with random snippets of background noise. Samples that are too
# long are truncated at the end, and samples that are too short are padded with
# 0s (also at the end). Augmented samples are stored in the specified directory.
# Note that all audio is mixed to mono.
#
# You will need the following packages (install via pip):
#  * numpy
#  * librosa
#  * soundfile
#  * shutil
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

import random
import argparse
from os import makedirs, listdir
from os.path import isdir, join, exists

import shutil
import librosa
import numpy as np
import soundfile as sf

# Specify version of this tool
__version__ = "0.1"

################################################################################
# Functions

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Taken from: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

# Print iterations progress
# From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Mix audio and random snippet of background noise
def mix_audio(word_path=None, 
                bg_path=None, 
                word_vol=1.0, 
                bg_vol=1.0, 
                sample_time=1.0,
                sample_rate=16000):
    
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
    waveform = [0.5 * word_vol * i for i in waveform] + (0.5 * bg_vol * bg_waveform[start_point:end_point])

    return waveform

################################################################################
# Main

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
                    help="Sample rate (Hz) of each output clip (default: 16000")
parser.add_argument('-e',
                    '--bit_depth',
                    action='store',
                    dest='bit_depth',
                    type=str,
                    choices=['PCM_16', 'PCM_24', 'PCM_32', 'PCM_U8', 'FLOAT',
                             'DOUBLE'],
                    default='PCM_16',
                    help="Bit depth of each sample (default: PCM_16)")

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

# Print tool welcome message
print("-----------------------------------------------------------------------")
print("Wake Word Mixer Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")

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
    resp = query_yes_no("Continue?")
    if resp:
        print("Deleting and recreating output directory.")
        shutil.rmtree(out_dir)
    else:
        print("Please delete directory to continue. Exiting.")
        exit()

# Create output dir
if not exists(out_dir):
    makedirs(out_dir)
# else:
#     print("ERROR: Output directory could not be deleted. Exiting.")
#     exit()

# Start with target words
for target in target_list:

    # Create subdirectory with the target word's name
    out_subdir = join(out_dir, target)
    makedirs(out_subdir)

    # Figure out number of digits so the filenames line up in OS
    word_path = join(words_dir, target_list[0])
    num_word_files = len(listdir(word_path))
    num_bg_files = len(listdir(bg_dir))
    num_output_files = num_word_files * num_bg_files
    digit_cnt = 1
    calc_num = num_output_files
    while(calc_num > 1):
        calc_num = calc_num // 10
        digit_cnt += 1

    # Show progress bar
    print("Mixing: '" + str(target) + "' (" + str(num_word_files) + 
            " word files * " + str(num_bg_files) + " bg files = " +
            str(num_output_files) + " output files)")
    printProgressBar(0, 
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
            filename = str(file_cnt).zfill(digit_cnt) + '.wav'
            sf.write(join(out_subdir, filename), 
                    waveform, 
                    sample_rate, 
                    subtype=bit_depth)
            file_cnt += 1

            # Update progress bar
            printProgressBar(file_cnt + 1, 
                            num_output_files, 
                            prefix="Progress:", 
                            suffix="Complete", 
                            length=50)

# Open one word and mix it

# word_files = [name for name in listdir(word_path)]
# bg_files = [name for name in listdir(bg_dir)]
# waveform = mix_audio(word_path=join(word_path, word_files[0]), 
#                     bg_path=join(bg_dir, bg_files[0]), 
#                     word_vol=word_vol, 
#                     bg_vol=bg_vol, 
#                     sample_time=sample_time,
#                     sample_rate=sample_rate)

# Save file
# filename = "test.wav"
# sf.write(join(out_dir, filename), waveform, sample_rate, subtype=bit_depth)