#!/usr/bin/env python3

"""
STFT Extraction

Script that reads in audio samples, resamples them, and computes the Short-Time
Fourier Transform (STFT) for each sample. The STFT (feature set) is computed via
the following steps:
1. Read in .wav file, resample and pad with 0s or truncate as needed
2. Get time slice of part of audio signal
3. Pad 0s to and of time slice if not long enough
3. Apply Hanning window to time slice
4. Find Fast Fourier Transform (FFT) of windowed time slice
5. Throw away frequency bins we don't care about (band-pass filter)
6. Scale FFT to mimic 16-bit fixed-point FFT
7. Average bins together to reduce number of bins in FFT
8. Map each bin's value to 8-bit number [0..255]
9. Append FFT slice to STFT array
10. Repeat steps 2-9 to create STFT that looks like a grayscale image

Note that all audio is mixed to mono.

Outputs are stored as Numpy arrays in a .npy file with the name corresponding to
the folder of the original .wav files. For example, if your .wav files are
stored in the directory hello, then the output features will be stored in 
hello.npy.

You will need the following packages (install via pip):
 * numpy
 * librosa
 * shutil

Example call:
python stft_extraction.py -d "../../../Python/datasets/custom_wake_words_mixed" 
            -o "../../../Python/datasets/custom_wake_words_features" 
            -s 1.0 -r 8000 -n 512 -i 400 -c 4000 -a 8 -b 3

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
import argparse
from os import makedirs, listdir, rename
from os.path import isdir, join, exists

import shutil
import librosa
import numpy as np

import utils

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2020, Shawn Hymel"
__license__ = "MIT"
__version__ = "0.1"

# Settings
stft_min_bin = 1        # Lowest bin to use (inclusive; filter out DC)

################################################################################
# Functions

# Compute compressed version of Short-Time Fourier Transform (STFT)
# TODO: Possible areas for improvement
#  - Use Mel-spaced frequency banks: https://stackoverflow.com/questions/54432486/normalizing-mel-spectrogram-to-unit-peak-amplitude
#  - Automatic gain control (normalize FFT before mapping to 8-bit value?)
def calc_stft(  waveform, 
                fs, 
                nfft,
                nhop, 
                min_bin, 
                max_bin,
                num_avg_bins,
                nslices,
                shift_bits):
    """
    Compute Short-Time Fourier Transform (STFT) of waveform. Remove first bin
    (DC) component and any other bins above given amount, creating a band-pass
    filter. Bins are averaged together to compress the size of the STFT. Each
    bin is mapped to 8-bit [0..255] value to create a small, grayscale image
    representing final STFT.
    @params:
        waveform    - Required  : Raw audio signal array (Float)
        fs          - Required  : Sampling rate of audio signal (Int)
        nfft        - Required  : Number of FFT points (Int)
        nhop        - Required  : Number of audio samples between FFTs (Int)
        min_bin     - Required  : Starting bin number to keep (Int)
        max_bin     - Required  : Ending bin number to keep (Int)
        num_avg_bins- Required  : How many bins to average together (Int)
        nslices     - Required  : Number of FFTs in each STFT (Int)
        shift_bits  - Required  : Number of bits to shift each bin value (Int)
    """

    # Create hanning window and empty STFT buffer
    hann_window = np.hanning(nfft)
    stft = np.zeros(((max_bin - min_bin) // num_avg_bins, 
                    nslices))

    # Find FFT of each windowed slice and append it to the STFT
    for i in range(stft.shape[1]):
        
        # Get window start and stop positions
        win_start = i * nhop
        win_stop = (i * nhop) + nfft

         # Pad 0s if window isn't long enough
        window = waveform[win_start:win_stop]
        if len(window) < nfft:
            window = np.append(window, np.zeros((nfft - len(window), 1)))

        # Apply hanning window and find FFT
        window = hann_window * window
        fft = np.abs(np.fft.rfft(window, n=nfft))

        # Only keep the frequency bins we care about (i.e. filter out unwanted 
        # frequencies)
        fft = fft[min_bin:max_bin] # With fs=8kHz, Nyquist is 4kHz

        # Adjust for quantization and scaling in 16-bit fixed point FFT
        fft = np.around(fft / nfft)

        # Average every <num_avg_bins> bins together to reduce size of FFT
        fft = np.mean(fft.reshape(-1, num_avg_bins), axis=1)

        # Reduce precision by converting to 8-bit unsigned values [0..255]
        fft = np.around(fft / (2 ** shift_bits))
        fft = np.clip(fft, a_min=0, a_max=255)

        # Put FFT slice into STFT
        stft[:, i] = fft

    return stft

################################################################################
# Main

###
# Parse command line arguments

# Script arguments
parser = argparse.ArgumentParser(description="Feature extraction tool that "
                                "computes the STFT of each audio file found.")
parser.add_argument('-d', 
                    '--samples_dir', 
                    action='store', 
                    dest='samples_dir',
                    type=str,
                    required=True,
                    help="Directory where the audio samples are stored. "
                        "Subdirectories in this directory should be named "
                        "after the different classes you wish to identify. ")
parser.add_argument('-o',
                   '--out_dir',
                   action='store',
                   dest='out_dir',
                   type=str,
                   required=True,
                   help="Directory where the feature set arrays are to be "
                        "stored")
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
                    default=8000,
                    help="Sample rate (Hz) of each output clip (default: "
                            "8000)")
parser.add_argument('-n',
                    '--nfft',
                    action='store',
                    dest='nfft',
                    type=int,
                    default=512,
                    help="Number of points in FFT calculation (default: 512)")
parser.add_argument('-i',
                    '--hop',
                    action='store',
                    dest='hop',
                    type=int,
                    default=400,
                    help="Number of audio samples between start of each FFT "
                            "slice (default: 400)")
parser.add_argument('-c',
                    '--cutoff_freq',
                    action='store',
                    dest='cutoff_freq',
                    type=int,
                    default=4000,
                    help="Drop FFT bins above this frequency (Hz) (default: "
                            "4000)")
parser.add_argument('-a',
                    '--num_avg_bins',
                    action='store',
                    dest='num_avg_bins',
                    type=int,
                    default=4000,
                    help="Number of bins to average together in FFT (default: "
                            "8)")
parser.add_argument('-b',
                    '--num_shift_bits',
                    action='store',
                    dest='num_shift_bits',
                    type=int,
                    default=3,
                    help="Number of bits to shift each FFT bin to make 8-bit "
                            "value, i.e. divide value by 2^(this_number) "
                            "(default: 3)")

# Parse arguments
args = parser.parse_args()
samples_dir = args.samples_dir
out_dir = args.out_dir
sample_time = args.sample_time
sample_rate = args.sample_rate
nfft = args.nfft
nhop = args.hop
cutoff_freq = args.cutoff_freq
num_avg_bins = args.num_avg_bins
num_shift_bits = args.num_shift_bits

# Calculated parameters
stft_n_slices = int(math.ceil(((sample_time * sample_rate) / nhop) - 
                (nfft / nhop)) + 1)
stft_max_bin = int((nfft / 2) / ((sample_rate / 2) / cutoff_freq)) + 1

###
# Welcome screen

# Print tool welcome message
print("-----------------------------------------------------------------------")
print("Wake Word Feature Extraction Tool")
print("v" + __version__)
print("-----------------------------------------------------------------------")

###
# Set up directories

# Create a list of possible words from the subdirectories
class_list = [name for name in listdir(samples_dir)]
print("Number of classes found:", len(class_list))

# Create output dir
if not exists(out_dir):
    makedirs(out_dir)

# Read the audio files in each class directory
for class_name in class_list:

    # Get number of files in each class and initialize sample set array
    class_dir = join(samples_dir, class_name)
    num_files = len(listdir(class_dir))
    sample_set = np.zeros(  (num_files, 
                            (stft_max_bin - stft_min_bin) // num_avg_bins, 
                            stft_n_slices))

    # Show progress bar
    print("Extracting features for " + class_name)
    utils.print_progress_bar(   0, 
                                num_files, 
                                prefix="Progress:", 
                                suffix="Complete", 
                                length=50)

    for i, filename in enumerate(listdir(class_dir)):

        # Open wav file, resample, mix to mono
        file_path = join(class_dir, filename)
        waveform, fs = librosa.load(file_path, sr=sample_rate, mono=True)

        # Convert floating point wav data (-1.0 to 1.0) to 16-bit PCM
        waveform = np.around(waveform * 32767)

        # Pad 0s on the end if not long enough
        if len(waveform) < sample_time * sample_rate:
            waveform = np.append(waveform, np.zeros(int((sample_time * 
                sample_rate) - len(waveform))))

        # Truncate if too long
        waveform = waveform[:int(sample_time * sample_rate)]

        # Get STFT for this one audio signal
        stft = calc_stft(   waveform, 
                            fs, 
                            nfft, 
                            nhop,
                            stft_min_bin, 
                            stft_max_bin, 
                            num_avg_bins,
                            stft_n_slices,
                            num_shift_bits)

        # Put STFT in feature sample set
        sample_set[i] = stft

        # Update progress bar
        utils.print_progress_bar(   i + 1, 
                                    num_files, 
                                    prefix="Progress:", 
                                    suffix="Complete", 
                                    length=50)
        
    # Print newline as workaround for progress bar covering up characters
    print()

    # Save tool version and sample set to file
    out_path = join(out_dir, class_name + ".npz")
    np.savez(out_path, version=__version__, samples=sample_set)

# Say we're done
print("Done!")
exit()