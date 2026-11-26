import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt

def name_visual():
    # --- Load audio and convert stereo to mono ---
    sample_rate, audio = wavfile.read("Titanic_2.wav")
    x_untrimmed = np.mean(audio, axis=1)

    # --- Trim silence at start and end ---
    threshold = 0.02 * np.max(np.abs(x_untrimmed))
    active = np.abs(x_untrimmed) > threshold

    if np.any(active):
        start = np.argmax(active)
        end = len(active) - np.argmax(active[::-1])
        x = x_untrimmed[start:end]
    else:
        x = x_untrimmed

    # --- Envelope via Hilbert transform ---
    envelope = np.abs(hilbert(x))

    # --- Smooth envelope with zero-phase low-pass filter ---
    cutoff = 20  # Hz
    filter_order = 3
    b, a = butter(filter_order, cutoff / (sample_rate / 2), btype="low")
    smooth_envelope = filtfilt(b, a, envelope)

    # --- Resample envelope at a coarse time step ---
    time_original = np.arange(len(smooth_envelope)) / sample_rate
    fs_envelope = 0.035  # seconds between samples

    time_sampled = np.arange(time_original[0], time_original[-1], fs_envelope)
    sampled_envelope = np.interp(time_sampled, time_original, smooth_envelope)

    # --- Plot: mirrored bars ---
    plt.figure(figsize=(12, 5))

    time_centered = time_sampled - time_sampled[-1] / 2
    bar_width = 0.03  # seconds

    plt.bar(time_centered, sampled_envelope, width=bar_width, color="blue")
    plt.bar(time_centered, -sampled_envelope, width=bar_width, color="blue")
    plt.axis("off")
    plt.tight_layout()
    plt.title("Titanic 2 visualized")
    plt.show()

#name_visual()