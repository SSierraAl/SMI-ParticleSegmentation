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
    n = len(datos)
    datos = np.asarray(datos, dtype=np.float64)
    fft_result = np.fft.rfft(datos)
    freq_fft = np.fft.rfftfreq(len(datos), 1 / samplefreq)
    amplitude = np.abs(fft_result) / n  # Normalize amplitude
    phase = np.angle(fft_result)
    return amplitude, freq_fft, phase

def multi_particle_signal(n_particles, v_range, m_range, theta, laser_lambda, po, time_max, adq_freq):
    """
    Simulate the signal from multiple particles with random speeds and modulation indices.

    Parameters
    -----------
    n_particles : int
        Number of particles.
    v_range : tuple
        Range of particle speeds (v_min, v_max) [m/s].
    m_range : tuple
        Range of modulation indices (m_min, m_max).
    theta : float
        Incident angle [degrees].
    laser_lambda : float
        Laser wavelength [m].
    po : float
        Laser power [mV].
    time_max : int
        Number of samples.
    adq_freq : float
        Acquisition frequency [Hz].

    Returns
    -----------
    P_f, t, f_D, m_i : Signal vector [mV], Time vector [s], Doppler frequencies [Hz], Modulation indices
    """
    # Convert angle to radians
    theta_rad = np.radians(90 - theta)  # Use 90 - theta for Doppler formula

    # Time vector
    t = np.linspace(0, time_max / adq_freq, time_max)

    # Generate random speeds and modulation indices
    v_min, v_max = v_range
    m_min, m_max = m_range
    v_i = np.random.uniform(v_min, v_max, n_particles)
    m_i = np.random.uniform(m_min, m_max, n_particles)

    # Calculate Doppler frequencies: f_D = 2 * v * sin(theta) / lambda
    f_D = (2 * v_i * np.sin(theta_rad)) / laser_lambda
    print('Doppler frequencies (kHz):', np.round(f_D / 1000, 2))
    print('Modulation indices:', np.round(m_i, 2))

    # Signal: P_f(t) = P_0 * [1 + sum_i m_i * cos(2 * pi * f_Di * t)]
    modulation = 1 + sum(m_i[j] * np.cos(2 * np.pi * f_D[j] * t) for j in range(n_particles))
    P_f = po * modulation

    return P_f, t, f_D, m_i

# Define parameters
n_particles = 10000                # Number of particles
v_range = (0, 0.25)         # Speed range [m/s]
m_range = (6,10)             # Modulation index range
theta = 80                  # Incident angle [degrees]
laser_lambda = 1550e-9        # Laser wavelength [m]
po = 0.004134 * 4             # Laser power [mV]
time_max = 10000               # Number of samples
adq_freq = 2e6                # Acquisition frequency [Hz]

# Generate the signal
P_f, t, f_D, m_i = multi_particle_signal(
    n_particles=n_particles,
    v_range=v_range,
    m_range=m_range,
    theta=theta,
    laser_lambda=laser_lambda,
    po=po,
    time_max=time_max,
    adq_freq=adq_freq
)

# Perform FFT
amplitude, freq_fft, phase = FFT_calc(P_f, adq_freq)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
plt.subplots_adjust(left=0.15, right=0.95, hspace=0.4)

# Plot time-domain signal
ax1.plot(t * 1000, P_f, label=r'$P_f(t)$', color='blue', linewidth=2)  # Time in ms
ax1.set_xlabel('Time [ms]', fontsize=30)
ax1.set_ylabel('Power [mV]', fontsize=30)
ax1.set_title('Time-Domain Signal', fontsize=32, pad=15)
ax1.tick_params(axis='both', which='major', labelsize=30, width=2, length=6)
ax1.legend(fontsize=30, loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot frequency-domain signal (amplitude spectrum)
ax2.plot(freq_fft / 1000, amplitude, label='FFT', color='blue', linewidth=2)  # Frequency in kHz
ax2.set_xlabel('Frequency [kHz]', fontsize=30)
ax2.set_ylabel('Amplitude [mV]', fontsize=30)
ax2.set_title('Frequency-Domain Signal', fontsize=32, pad=15)
ax2.tick_params(axis='both', which='major', labelsize=32, width=2, length=6)
ax2.legend(fontsize=32, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.7)
#ax2.set_xlim(0, 500)  # Limit to 30 kHz to focus on Doppler frequencies


plt.show()