import numpy as np
import scipy.signal as signal

def Downsample(x, fs_in, fs_out):
  assert np.isclose(fs_in / fs_out, int(fs_in / fs_out))

  # Filter
  out_nyquist = fs_out / 2
  fc = 0.9 * out_nyquist
  

  numtaps = 513 # Determined Experimentally
  
  bb = signal.firwin(numtaps,fc, fs=fs_in)
  y = signal.oaconvolve(x, bb)

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