# You'll probably need PortAudio:
#   sudo apt install libportaudio2

import sounddevice as sd
import numpy as np 
import timeit
from scipy import signal
from tflite_runtime.interpreter import Interpreter
import math

# Settings
model_path = 'goodnight_how_are_you_model.tflite'
labels = ['goodnight', 'how_are_you', 'other', 'background'] # Always include 'other' and 'background'
sample_time = 1.0       # Time for 1 sample (sec)
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
ffts_per_inference = 2  # Number of FFTs to compute before performing inference
maf_pts = 3             # Number of points (depth) in moving average filter
threshold = 0.5         # Softmax value for target word needs to be over this
holdoff_time = 1.5      # Time (sec) before another target word can trigger

# Calculated parameters
stft_n_slices = int(math.ceil(((sample_time * resample_rate) / stft_n_hop) - 
                (stft_n_fft / stft_n_hop)) + 1)
stft_max_bin = int((stft_n_fft / 2) / ((resample_rate / 2) / filter_cutoff)) + 1
stft_n_bins = (stft_max_bin - stft_min_bin) // stft_avg_bins
stft_n_overlap = stft_n_fft - stft_n_hop
hann_window = np.hanning(stft_n_fft)
holdoff_stfts = int(holdoff_time / (ffts_per_inference * 
                (stft_n_hop / resample_rate)))
num_targets = len(labels) - 2
print('N slices:', stft_n_slices)
print('STFT max bin:', stft_max_bin)
print('Num STFTs for holdoff:', holdoff_stfts)

# Some global variables to share with our audio callback
g_flag = 0
g_audio_buf = np.zeros((stft_n_fft,))

# Create our STFT buffer
stft = np.zeros((stft_n_bins, stft_n_slices))
print('STFT shape:', stft.shape)

# Some other global variables not used by the callback
stft_shift = stft.shape[1] - 1      # Shift STFT over by 1 time slice
fft_cnt = 0                         # Count number of FFTs computed
maf_buf = np.zeros((maf_pts, num_targets))  # Moving average filter buffer
stfts_cnt = 0                       # Count number of STFTs computed
in_holdoff = 0                      # Remember if we're in holdoff period

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Resample
def resample(sig, old_fs, new_fs):
    seconds = len(sig) / old_fs
    num_samples = seconds * new_fs
    resampled_signal = signal.resample(sig, int(num_samples))
    #signal = signal / np.iinfo(np.int16).max
    return resampled_signal

# This gets called every time blocksize fills up
def sd_callback(rec, frames, time, status):

    # Declare global variables
    global g_flag
    global g_audio_buf
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec = resample(rec, sample_rate, resample_rate)

    # Convert floating point wav data (-1.0 to 1.0) to 16-bit PCM
    rec = np.around(rec * 32767)

    # Copy last <n_overlap> samples to front of buffer
    g_audio_buf[:stft_n_overlap] = g_audio_buf[-stft_n_overlap:]

    # Append new recording to end of buffer
    g_audio_buf[stft_n_overlap:] = rec

    # Test
    g_flag = 1

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int((sample_rate / resample_rate) * stft_n_hop),
                    callback=sd_callback):

    start = timeit.default_timer()
    while True:
        
        # If we get a flag from interrupt, compute features
        if g_flag == 1:
            start = timeit.default_timer()
            
            # Get a window and reset flag
            window = hann_window * g_audio_buf
            g_flag = 0

            # Calculate FFT
            fft = np.abs(np.fft.rfft(window, n=stft_n_fft))

            # Only keep the frequency bins we care about (i.e. filter out unwanted frequencies)
            fft = fft[stft_min_bin:stft_max_bin] # With fs=8kHz, Nyquist is 4kHz

            # Adjust for quantization and scaling in 16-bit fixed point FFT
            fft = np.around(fft / stft_n_fft)

            # Average every <stft_avg_bins bins> together to reduce size of FFT
            fft = np.mean(fft.reshape(-1, 8), axis=1)

            # Reduce precision by converting to 8-bit unsigned values [0..255]
            fft = np.around(fft / (2 ** shift_n_bits))
            fft = np.clip(fft, a_min=0, a_max=255)

            # Shift STFT to the left by one slice in time and insert FFT at end
            stft[:,:stft_shift] = stft[:,-stft_shift:]
            stft[:,-1] = fft
            
            # Every time we calculate x FFTs, perform an inference with STFT
            fft_cnt += 1
            if fft_cnt >= ffts_per_inference:
                fft_cnt = 0

                # Don't do inference if we're in a holdoff period
                if in_holdoff == 1:
                    stfts_cnt += 1
                    if stfts_cnt == holdoff_stfts:
                        in_holdoff = 0
                        stfts_cnt = 0
                else:

                    # Reshape features
                    in_tensor = np.float32(stft.reshape(1, stft.shape[0], stft.shape[1]))
                    interpreter.set_tensor(input_details[0]['index'], in_tensor)
                    
                    # Infer!
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    val = output_data[0]

                    # Push target word inference output values to buffer
                    maf_buf[:-1] = maf_buf[1:]
                    maf_buf[-1] = val[:-2] # Last 2 entries are 'other' and 'bg'

                    # Get average in each target category
                    maf_avg = np.sum(maf_buf, axis=0) / maf_pts
                    
                    # Print out result
                    print(maf_avg)
                    max_idx = np.argmax(maf_avg)
                    if maf_avg[max_idx] >= threshold:
                        print(labels[max_idx])
                        in_holdoff = 1

            #print('Time (ms):', timeit.default_timer() - start)
            
        pass