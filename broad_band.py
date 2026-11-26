import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def get_broad_band_noise(x, N=4, Wn1=20, Wn2=22000, fs=44100, perturb=0.0, plot=True, seed=0):
    """
    x        - input signal
    N        - filter order
    Wn1, Wn2 - band edges in Hz (0 < Wn1 < Wn2 < fs/2)
    fs       - sampling frequency
    perturb  - relative std of random noise on the SOS coefficients (0 = none)
    plot     - show the frequency response
    """
    sos = signal.butter(N, [Wn1, Wn2], btype="bandpass", fs=fs, output="sos")

    rng = np.random.default_rng(seed)
    sos_noise = sos * (1 + rng.normal(0, perturb, size=sos.shape))
    sos_noise[:, 3] = 1.0  # a0 must stay exactly 1

    # Stability check
    poles = np.concatenate([np.roots(s[3:]) for s in sos_noise])
    if np.abs(poles).max() >= 1:
        raise ValueError("Perturbed filter is unstable, lower `perturb`")
    
    if plot:
        # Plot broad band
        w0, h0 = signal.sosfreqz(sos, worN=4096, fs=fs)
        w, h = signal.sosfreqz(sos_noise, worN=4096, fs=fs)
        plt.plot(w0, 20 * np.log10(np.maximum(np.abs(h0), 1e-8)), label="original")
        plt.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-8)), label="perturbed")
        plt.axvspan(Wn1, Wn2, alpha=0.2)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Gain [dB]")
        plt.grid(True)
        plt.title('Broad band')
        plt.legend()
        plt.show()

        # Plot power spectral density using Welch’s method
        f, Pxx_den = signal.welch(w0, fs=fs)
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency')
        plt.ylabel('PSD')
        plt.grid()
        plt.title('PSD using Welch’s method')
        plt.show()

    return signal.sosfiltfilt(sos_noise, x)
