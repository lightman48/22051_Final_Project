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

def ButterworthBandpass(x: np.ndarray, wl: float, wh: float, order: float, samplerate):
  sos = signal.butter(order, [wl, wh], fs=samplerate, btype="bandpass", output="sos")
  y = signal.sosfilt(sos, x)
  return y

def GenerateBandlimitedNoise(T: float, fs: float, fbounds: list, order = 6):
    """
    T= duration in seconds
    fs= sampling frequency
    returns a vector of noise to be added to sound file
    """
    x = np.random.rand(int(T*fs), dtype=np.float32)

    sos = signal.cheby2(order,40, fbounds, btype="bandpass", fs=fs, output="sos")
    noise = signal.sosfilt(sos,x)
    return noise