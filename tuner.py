import sympy as sp
sp.init_printing()
import scipy.fft as fft
import scipy.signal as sig
import numpy as np
import scipy as sc
from scipy.io import wavfile
import matplotlib.pyplot as plt 
import scipy.fft as fft
from scipy.spatial import distance

def tuner(wave,srate):
    if wave.ndim > 1: # Converts to mono by removing a channel, if applicable 
        wave = wave[:, 0]
    time_axis = np.arange(len(wave)) / srate
    wave = wave - np.mean(wave) # Remove dc offset

    mag_wave = np.abs(fft.fft(wave)) # Get magnitude of each freq

    freq_axis = fft.fftfreq(len(wave), d=1 / srate) # Special fft that orders the results acroding to thier physical freq.
    half_len = len(wave) // 2 

    pos_freqs = freq_axis[:half_len] # cut away the mirroed parts
    pos_amplitudes = mag_wave[:half_len] # ditto
    dominant_freq = pos_freqs[np.argmax(pos_amplitudes)] #np.argmax returns index of highest freq in pos_amplitudes
    unfilt_env = np.abs(sig.hilbert(wave))

    window_length = int(0.005 * srate) # window for the moving avrage filter, looking at a window of 5 miliseconds
    env = np.convolve(unfilt_env, np.ones(window_length)/window_length, mode='same') # These two lines apply a moving avrage filter to the evolope, to ensure a single random nosiy elements and overtones dosnt get picked as the loudest freq

    max_env = np.max(env)
    peak_idx = np.argmax(env)
    peak_time = time_axis[peak_idx]

    start_lim = 0.20 * np.max(env) # This limit is arbitrary and is there to ensure we only count the time of the note and not before the note is played
    start_idx = np.where(env > start_lim)[0][0] # Finds the index of the first place the value of the envolope is grater than the limit
    attack = (peak_idx - start_idx) / srate # This is done to find the duration in seconds rather than in samples.

    sliced_env = env[peak_idx:] # starting at the peak to ensure its only the note we are looking at.
    decay_lim = 0.70 * np.max(env) # same here arbitrary, or well guesstimated
    decay_idx = np.where(sliced_env < decay_lim)[0][0] # Finds the index of the first place its smaller
    decay = decay_idx / srate

    noise_floor = 0.05 * np.max(env)
    sustain_start = 0.70 * np.max(env)

    end_idx = np.where(env > noise_floor)[0][-1] # takes the last place the envolope is over the noisefloor
    rel_idx = np.where(env > sustain_start)[0][-1] # takes the last place it is over the start for the sustain. This way we look at the period after the decay but before it fades into nothing
    release = (end_idx - rel_idx) / srate

    decay_end_idx = peak_idx + decay_idx
    sustain = (rel_idx - decay_end_idx) / srate
    num_harmonics = 4 # Looking at the first 4 harmonics as this should be suffecient
    harmonic_amps = [] 
    inharmonicity_devi = [] 

    for i in range(1, num_harmonics + 1):
        target_freq = dominant_freq * i
        close_idx = np.argmin(np.abs(pos_freqs - target_freq)) # This way we find the index clossest to the harmonic we are cheking for

        window = 5 # We cant expect our recordings to behave quite as nicely as the maths would imply, we therefore check in a little area around the overtone we are looking at, to ensure we dont miss it in case it is a little off
        win_amps = pos_amplitudes[close_idx - window : close_idx + window + 1]
        win_freqs = pos_freqs[close_idx - window : close_idx + window + 1]

        local_peak_idx = np.argmax(win_amps)
        local_peak_amp = win_amps[local_peak_idx]
        local_peak_freq = win_freqs[local_peak_idx]
        harmonic_amps.append(local_peak_amp)

        if i > 1: # only doing it from the first harmonic onwards as the f0 will always have a deviation of 0 as the sound file dictates where that is
            inharmonicity_devi.append(abs(local_peak_freq - target_freq)) # checks the disance from the harmonic to the mathmatically perfect harmoinc

    norm_harmonics = np.array(harmonic_amps) / harmonic_amps[0] # normalize to ensure that the volume the intrument is playing dosnt influence its detection
    avg_inharmonicity = np.mean(inharmonicity_devi)
    spectral_centroid = np.sum(pos_freqs * pos_amplitudes) / np.sum(pos_amplitudes) # This finds the "center of mass" for the magnitude of the frequancies so to speak

    gamma = np.array([
        dominant_freq,         
        spectral_centroid,     
        avg_inharmonicity,    
        attack,       
        decay,        
        norm_harmonics[1], # second, thrid, and so on harmonic 
        norm_harmonics[2],
        norm_harmonics[3],
    # norm_harmonics[4],
    # norm_harmonics[5],
    ])
    gamma_database = { # this is a dictionary made up of gamma arrays gotten from earlier runs of my program, with known samples
    "Trumpet": np.array([9.93666667e+02, 1.82548312e+03, 1.22222222e+00, 5.86961451e-01,
       1.24716553e-03, 4.09611776e-01, 5.75753294e-02, 1.45392826e-02]),
    "Flute": np.array([4.44771803e+02, 2.13551793e+03, 8.22281019e-02, 3.03011338e+00,
       4.80272109e-01, 4.50092176e-01, 9.55729115e-02, 3.38497282e-01]),
    "Piano": np.array([4.40562106e+02, 1.60550605e+03, 1.00051954e-01, 8.70748299e-03,
       1.40589569e-02, 4.11730613e-02, 1.75579779e-02, 1.62767149e-03]),
    "Guitar": np.array([2.63426655e+02, 3.04943249e+03, 6.39667373e-01, 1.23750000e-02,
       6.04166667e-03, 3.41547483e-01, 2.95447008e-01, 1.99964970e-01])
    }

    scales = np.array([2000, 10000, 10, 2, 2, 1, 1, 1]) # This is an array of expected max values for each thing we are comparing, used for nomilization so that a diviation of say 30 hz dosnt count more than a diffrence in attack of say 20 seconds. The values are chosen based on intuition from looking at gamma arrays
    norm_unknown = gamma/scales

    distances = {} # initiate list
    for intrument, ref_vector in gamma_database.items():
        norm_ref = ref_vector / scales
        distances[intrument] = distance.euclidean(norm_unknown, norm_ref) # This takes the L2 norm of all the diffences and puts them into a list

    best_match = min(distances, key=distances.get) # picks the best out of the bunch
    return best_match