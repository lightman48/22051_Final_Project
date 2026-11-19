from scipy import signal
import numpy as np
def narrowband_noise(T, fs):
    """
    T= duration in seconds
    fs= sampling frequency
    returns a vector of noise to be added to sound file
    """
    x = np.random.randn(T*fs)

    b, a = signal.cheby2(
        6,
        40,
        [6000,8000],
        btype="bandpass",
        fs=fs
)
    noise = signal.lfilter(b,a,x)
    return noise