
import numpy as np
import matplotlib.pyplot as plt

# Set Matplotlib parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

def FFT_calc(datos, samplefreq):
    """
    Calculate the FFT (Fast Fourier Transform) of the input data.
    
    Parameters:
    datos : array-like
        Time-domain signal data.
    samplefreq : int
        Sampling frequency of the data in Hz.

    Returns:
    amplitude : array-like
        Amplitude spectrum of the signal.
    freq_fft : array-like
        Corresponding frequency values.
    phase : array-like
        Phase spectrum of the signal.
    """
    n = len(datos)  # Length of the data
    datos = np.asarray(datos, dtype=np.float64)
    fft_result = np.fft.rfft(datos)  # Compute FFT
    freq_fft = np.fft.rfftfreq(len(datos), 1 / samplefreq)  # Corresponding frequencies
    amplitude = np.abs(fft_result) / n  # Normalize amplitude
    phase = np.angle(fft_result)  # Phase of the FFT
    return amplitude, freq_fft, phase

def simulated_particle(P_size, P_Speed, Inc_Angle, Laser_Lambda, Po, T_impact, S_l, Time_max, Adq_Freq, m0):
    """
    Simulate the signal of a single particle based on Optical Feedback Interferometry.

    Parameters
    -----------
    P_size      : Diameter of the particle [m]
    P_Speed     : Particle speed [m/s]
    Inc_Angle   : Incident angle [degrees]
    Laser_Lambda: Laser wavelength [m]
    Po          : Laser power [mV]
    T_impact    : Time when particle center crosses beam center [s]
    S_l         : Laser beam spot diameter [m]
    Time_max    : Number of samples
    Adq_Freq    : Acquisition frequency [Hz]
    m0          : Modulation index

    Returns
    -----------
    P_t, t      : Signal vector [mV], Time vector [s]
    """
    # Convert angle from degrees to radians
    Inc_Angle_rad = np.radians(Inc_Angle)
    Inc_Angle_rad_2 = np.radians(90 - Inc_Angle)

    # Time vector in seconds
    t = np.linspace(0, Time_max / Adq_Freq, Time_max)

    # Doppler frequency: f_D = 2 * V * sin(theta) / lambda
    f_D = (2 * P_Speed * np.sin(Inc_Angle_rad_2)) / Laser_Lambda
    print('Doppler frequency:', f_D, 'Hz')

    print('Particle Speed:', P_Speed, 'm/s')

    # Transit time
    tau = (P_size + S_l) / (P_Speed * np.sin(Inc_Angle_rad))
    print('Transit time (tau):', tau, 's')

    # Signal: P_f(t) = P_0 * [1 + m_0 * cos(2 * pi * f_D * t)] * exp(-((t - t_0)^2) / (2 * tau^2))
    modulation = 1 + m0 * np.cos(2 * np.pi * f_D * t)
    envelope = np.exp(-((t - T_impact)**2) / (2 * tau**2))
    P_t = Po * modulation * envelope

    return P_t, t

# Define parameters
D_Particle = 2e-6      # Particle diameter [m]
P_Speed = 0.108        # Particle speed [m/s], set for f_D ≈ 20 kHz
Theta = 80             # Incident angle [degrees]
Laser_lambda = 1550e-9 # Laser wavelength [m]
Po = 0.004134 * 4      # Laser power [mV]
T_impact = 0.0007      # Time of impact [s]
S_l = 0.7e-5            # Laser spot diameter [m]
Time_max = 3000        # Number of samples
Adq_Freq = 2e6         # Acquisition frequency [Hz]
M0_init = 11.3         # Initial modulation index 

# Generate the signal
P_t, t = simulated_particle(
    P_size=D_Particle,
    P_Speed=P_Speed,
    Inc_Angle=Theta,
    Laser_Lambda=Laser_lambda,
    Po=Po,
    T_impact=T_impact,
    S_l=S_l,
    Time_max=Time_max,
    Adq_Freq=Adq_Freq,
    m0=M0_init
)

# Perform FFT
amplitude, freq_fft, phase = FFT_calc(P_t, Adq_Freq)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharex=False)
plt.subplots_adjust(left=0.15, right=0.95, hspace=0.4)

# Plot time-domain signal
ax1.plot(t * 1000, P_t, label=r'$2 \mu m$', color='blue', linewidth=2)  # Time in ms
ax1.set_xlabel('Time [ms]', fontsize=30)
ax1.set_ylabel('Power [mV]', fontsize=30)
ax1.set_title('Time-domain Signal', fontsize=32, pad=15)
ax1.tick_params(axis='both', which='major', labelsize=30, width=2, length=6)
ax1.legend(fontsize=30)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot frequency-domain signal (amplitude spectrum)
ax2.plot(freq_fft / 1000, amplitude, label='FFT', color='blue', linewidth=2)  # Frequency in kHz
ax2.set_xlabel('Frequency [kHz]', fontsize=30)
ax2.set_ylabel('Amplitude [mV]', fontsize=30)
ax2.set_title('Frequency-domain Signal', fontsize=32, pad=15)
ax2.tick_params(axis='both', which='major', labelsize=30, width=2, length=6)
ax2.legend(fontsize=30)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(0, 50)  # Limit to 50 kHz for better visualization

plt.show()

