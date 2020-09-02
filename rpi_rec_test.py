#!/usr/bin/env python3

"""
Raspberry Pi Recording Test

Script that records a snippet of audio and prints out the raw audio (as 16-bit 
integers), STFT, or features (modified STFT).

You will likely need to install PortAudio:
    sudo apt install libportaudio2

You will need the following packages (install via pip)
 * numpy
 * scipy
 * sounddevice

Example call:
python rpi_rec_test.py -o "raw" -t 1.0

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

import numpy as np
import sounddevice as sd
from scipy import signal

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2020, Shawn Hymel"
__license__ = "MIT"
__version__ = "0.1"

# Settings
sample_rate = 48000     # Sample rate (Hz) of microphone
resample_rate = 8000    # Downsample to this rate (Hz)
filter_cutoff = 4000    # Remove frequencies above this threshold (Hz)
num_channels = 1
stft_n_fft = 512        # Number of FFT bins (also, number of samples in each slice)
stft_n_hop = 400        # Distance between start of each FFT slice (number of samples)
stft_window = 'hanning' # "The window of choice if you don't have any better ideas"
stft_min_bin = 1        # Lowest bin to use (inclusive; basically, filter out DC)       
stft_avg_bins = 8       # Number of bins to average together to reduce FFT size
shift_n_bits = 3        # Number of bits to shift 16-bit STFT values to make 8-bit values (before clipping)

################################################################################
# Functions

# Resample
def resample(sig, old_fs, new_fs):
    seconds = len(sig) / old_fs
    num_samples = seconds * new_fs
    resampled_signal = signal.resample(sig, int(num_samples))
    #signal = signal / np.iinfo(np.int16).max
    return resampled_signal

# Calculate STFT
def calc_stft(waveform, fs, nfft, nhop):
    return

################################################################################
# Main

###
# Parse command line arguments

# Script arguments
parser = argparse.ArgumentParser(description="Records -t seconds of audio and "
                                    "prints out raw audio (16-bit integers), "
                                    "STFT, or features (modified STFT).")
parser.add_argument('-o',
                    '--output',
                    action='store',
                    dest='output',
                    type=str,
                    default='raw',
                    choices=['raw', 'stft', 'features'],
                    help="Choose to output raw, stft, or features to console")
parser.add_argument('-t',
                    '--time',
                    action='store',
                    dest='time',
                    type=float,
                    default=1.0,
                    help="Time (in seconds) to record audio.")

# Parse arguments
args = parser.parse_args()
output_format = args.output
duration = args.time

# Wait for user to press enter to start recording
input("Press Enter to continue, recording starts on 'GO!'...")

# Record for set time
rec = sd.rec(int((1.0 + duration) * sample_rate), 
                samplerate=sample_rate, 
                channels=num_channels)
time.sleep(1.0)
print("GO!")
sd.wait()

# Remove first part of audio recording
rec = rec[int(1.0 * sample_rate):]

# Remove 2nd dimension from recording sample
rec = np.squeeze(rec)

# Resample
rec = resample(rec, sample_rate, resample_rate)

# Convert floating point wav data (-1.0 to 1.0) to 16-bit PCM
rec = np.around(rec * 32767)

# Print raw audio
if output_format == 'raw':
    for s in rec[:-1]:
        print(int(s), end=", ")
    print(int(rec[-1]))

# Print STFT
elif output_format == 'stft':
    print("Not implemented yet")

# Print features (modified STFT)
elif output_format == 'features':
    print("Not implemented yet")