import numpy as np
import scipy.signal as signal

def AntiAliasingFilter(fs_hi, fs_low, numtaps = 1023): # Numtaps Determined Experimentally
  # Filter
  low_nyquist = fs_low / 2
  fc = 0.9 * low_nyquist

  return signal.firwin(numtaps,fc, fs=fs_hi)

def Downsample(x, fs_in, fs_out) -> np.ndarray:
  assert np.isclose(fs_in / fs_out, int(fs_in / fs_out))

  numtaps = 1023
  filt = AntiAliasingFilter(fs_in, fs_out, numtaps)
  y = signal.oaconvolve(x, filt)

  delay_samples = (numtaps - 1) // 2
  y = y[delay_samples:]

  # memcpy
  outlen = int(len(x) * fs_out / fs_in)
  out_indices = np.arange(outlen)
  in_indices = out_indices * fs_in / fs_out
  out = np.interp(in_indices, np.arange(len(y)), y)
  
  return out

def gaindb(g):
  return 20*np.log10(g)