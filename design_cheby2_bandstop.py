import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
def design_cheby2_bandstop(fs, fp1, fp2, fs1, fs2,Rs):    
    """
    fs= sampling frequency
    fp1,fp2= passband edges (a little outside of stopband edges)
    fs1,fs2= stopband edges
    Rs= stopband attenuation
    returns a vector of noise to be added to sound file
    for our project: fp1 and fp2 could be 5500 and 8500, fs 6000 and 8000
    """
    T = 1/fs
    wp1 = fp1*2*np.pi/fs
    wp2 = fp2*2*np.pi/fs
    ws1 = fs1*2*np.pi/fs
    ws2 = fs2*2*np.pi/fs
    Omega_p1 = (2/T)*np.tan(wp1/2) #prewarping
    Omega_p2 = (2/T)*np.tan(wp2/2)
    Omega_s1 = (2/T)*np.tan(ws1/2)
    Omega_s2 = (2/T)*np.tan(ws2/2)
    Omega_0 = np.sqrt(Omega_s1*Omega_s2)
    B = Omega_s2-Omega_s1
    Omega_lp_p1 = abs(B*Omega_p1/(Omega_0**2 - Omega_p1**2))
    Omega_lp_p2 = abs(B*Omega_p2/(Omega_0**2 - Omega_p2**2))
    Omega_lp_s1 = abs(B*Omega_s1/(Omega_0**2 - Omega_s1**2))
    Omega_lp_s2 = abs(B*Omega_s2/(Omega_0**2 - Omega_s2**2))
    Omega_p = max(Omega_lp_p1, Omega_lp_p2)
    Omega_s = min(Omega_lp_s1, Omega_lp_s2)

    N, Wn = signal.cheb2ord(Omega_p, Omega_s, 1, Rs, analog=True)
    b,a = signal.cheby2(N,Rs,Wn,btype="low",analog=True)
    b_bs,a_bs = signal.lp2bs(b,a,wo=Omega_0,bw=B)
    bz,az = signal.bilinear(b_bs,a_bs,fs)
    
    w, H = signal.freqz(bz, az, worN=8192, fs=fs)

    plt.figure(figsize=(10,5))
    plt.plot(w, 20*np.log10(np.abs(H)))
    plt.grid(True)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Chebyshev II Bandstop Filter")
    plt.axvline(5500,color="g",linestyle="--")
    plt.axvline(6000,color="r",linestyle="--")
    plt.axvline(8000,color="r",linestyle="--")
    plt.axvline(8500,color="g",linestyle="--")
    plt.show()
    plt.figure(figsize=(10,5))
    plt.plot(w,np.unwrap(np.angle(H)))
    plt.grid(True)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")
    plt.title("Phase Response")
    plt.show()

    z,p,k = signal.tf2zpk(bz,az)
    theta = np.linspace(0,2*np.pi,1000)
    plt.figure(figsize=(6,6))
    plt.plot(np.cos(theta),np.sin(theta),'k--')
    plt.scatter(np.real(z),np.imag(z),marker='o',label='Zeros')
    plt.scatter(np.real(p),np.imag(p),marker='x',label='Poles')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Pole-Zero Map')
    plt.show()

    imp = np.zeros(500)
    imp[0] = 1
    h = signal.lfilter(bz,az,imp)
    plt.figure(figsize=(10,4))
    plt.stem(h)
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title('Impulse Response')
    plt.show()

    return bz, az